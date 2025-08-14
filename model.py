import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1)]


class PreNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.dropout = dropout

    def forward(self, x):
        # Keep dropout active even in eval, Tacotron-style
        self.net[2].train()
        self.net[5].train()
        return self.net(x)


class PostNet(nn.Module):
    """ Uses 5 conv layers with residuals """
    def __init__(
        self,
        n_mels: int,
        channels: int = 512,
        kernel_size: int = 5,
        dropout: float = 0.5
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2

        convs = []
        # First layer: n_mels -> channels
        convs.append(nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.Tanh(),
            nn.Dropout(dropout),
        ))
        # Middle layers: channels -> channels
        for _ in range(3):
            convs.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=padding),
                nn.BatchNorm1d(channels),
                nn.Tanh(),
                nn.Dropout(dropout),
            ))
        # Final layer: channels -> n_mels (no tanh)
        convs.append(nn.Sequential(
            nn.Conv1d(channels, n_mels, kernel_size, padding=padding),
            nn.BatchNorm1d(n_mels),
            nn.Dropout(dropout),
        ))
        self.convs = nn.ModuleList(convs)

    def forward(self, mel_pred_bt_f):  # [B, T, n_mels]
        x = mel_pred_bt_f.transpose(1, 2)  # [B, n_mels, T]
        for i, block in enumerate(self.convs):
            x = block(x)
        x = x.transpose(1, 2)  # [B, T, n_mels]
        return x


class TransformerTTS(nn.Module):
    """Autoregressive Transformer TTS with scheduled sampling support and optional PostNet."""

    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.text_embedding = nn.Embedding(hp.vocab_size, hp.d_model, padding_idx=hp.PAD)
        self.text_pos_encoding = PositionalEncoding(hp.d_model, hp.max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hp.d_model,
            nhead=hp.num_heads,
            dim_feedforward=hp.d_ff,
            dropout=hp.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, hp.num_layers)

        self.prenet = PreNet(
            in_dim=hp.n_mels,
            hidden_dim=hp.prenet_hidden,
            out_dim=hp.d_model,
            dropout=hp.prenet_dropout
        )
        self.mel_pos_encoding = PositionalEncoding(hp.d_model, hp.max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hp.d_model,
            nhead=hp.num_heads,
            dim_feedforward=hp.d_ff,
            dropout=hp.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, hp.num_layers)

        self.mel_linear = nn.Linear(hp.d_model, hp.n_mels)
        self.gate_linear = nn.Sequential(
            nn.Linear(hp.d_model, hp.gate_hidden),
            nn.ReLU(),
            nn.Dropout(hp.dropout),
            nn.Linear(hp.gate_hidden, 1)
        )

        # postnet 
        self.postnet = PostNet(n_mels=hp.n_mels, channels=hp.postnet_channels, kernel_size=hp.postnet_kernel_size, dropout=hp.postnet_dropout)
        self._postnet_enabled = False  # stage 1: False, stage 2: True

        self._init_weights()


    # stage control
    def set_postnet(self, enabled: bool):
        self._postnet_enabled = bool(enabled)

    def postnet_parameters(self):
        return list(self.postnet.parameters())

    def base_parameters(self):
        """All params except PostNet (encoder/decoder/heads)."""
        postnet_ids = set(id(p) for p in self.postnet.parameters())
        return [p for p in self.parameters() if id(p) not in postnet_ids]


    def _init_weights(self):
        """ xavier uniform """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Start pessimistic on gate prediction
        nn.init.constant_(self.gate_linear[-1].bias, self.hp.gate_bias_init)

    @staticmethod
    def _build_key_padding_mask(lengths, max_len, device):
        """
        lengths: List[int] or 1D tensor with batch lengths
        returns: Bool mask [B, T] where True indicates PAD positions.
        """
        if isinstance(lengths, int):
            lengths = [lengths]
        if torch.is_tensor(lengths):
            lengths = lengths.tolist()
        B = len(lengths)
        mask = torch.zeros((B, max_len), device=device, dtype=torch.bool)
        for i, L in enumerate(lengths):
            if L < max_len:
                mask[i, L:] = True
        return mask

    def _causal_mask(self, T, device):
        """Upper triangular mask [T, T] with True above diagonal"""
        return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)


    def forward(self, text, text_lens, mels, mel_lens, use_scheduled_sampling=False):
        """
        Forward pass with optional scheduled sampling

        Returns:
            mel_out: [B, T_mel, n_mels] (postnet if enabled, else decoder output)
            gate_out: [B, T_mel]
        """
        device = text.device
        B, T_text = text.size()
        T_mel = mels.size(1)

        
        src_pad = self._build_key_padding_mask(text_lens, T_text, device)   # [B, T_text]

        
        text_emb = self.text_embedding(text)
        text_emb = self.text_pos_encoding(text_emb)
        memory = self.encoder(text_emb, src_key_padding_mask=src_pad)

        if use_scheduled_sampling and self.training:
            mel_before, gate_out = self._forward_with_scheduled_sampling(
                memory, mels, mel_lens, src_pad, device
            )
        else:
            # full teacher forcing 
            tgt_pad = self._build_key_padding_mask(mel_lens, T_mel, device)     # [B, T_mel]
            tgt_causal = self._causal_mask(T_mel, device)                        # [T_mel, T_mel]

            mel_emb = self.prenet(mels)
            mel_emb = self.mel_pos_encoding(mel_emb)
            dec_out = self.decoder(
                mel_emb, memory,
                tgt_mask=tgt_causal,
                tgt_key_padding_mask=tgt_pad,
                memory_key_padding_mask=src_pad
            )

            # heads
            mel_before = self.mel_linear(dec_out)                # [B, T_mel, n_mels]
            gate_out = self.gate_linear(dec_out).squeeze(-1)     # [B, T_mel]

        # postnet
        if self._postnet_enabled:
            mel_out = mel_before + self.postnet(mel_before)
        else:
            mel_out = mel_before

        return mel_out, gate_out

    def _forward_with_scheduled_sampling(self, memory, mels, mel_lens, src_pad, device):
        """
        Memory-efficient scheduled sampling: two decoder passes
        1) Teacher forcing to get predictions (no grad)
        2) Mix ground truth with predictions as inputs (with grad)
        """
        B, T_mel, n_mels = mels.size()
        sampling_ratio = self.hp.scheduled_sampling_ratio

        # Pass 1 (no grad)
        tgt_pad = self._build_key_padding_mask(mel_lens, T_mel, device)
        tgt_causal = self._causal_mask(T_mel, device)

        with torch.no_grad():
            mel_emb = self.prenet(mels)
            mel_emb = self.mel_pos_encoding(mel_emb)
            dec_out = self.decoder(
                mel_emb, memory,
                tgt_mask=tgt_causal,
                tgt_key_padding_mask=tgt_pad,
                memory_key_padding_mask=src_pad
            )
            mel_pred = self.mel_linear(dec_out).detach()

        mixed_mels = mels.clone()
        for b in range(B):
            seq_len = int(mel_lens[b].item())
            for t in range(1, min(seq_len, T_mel)):
                # with probability (1 - sampling_ratio)
                if random.random() > sampling_ratio:
                    mixed_mels[b, t] = mel_pred[b, t - 1]

        # second pass 
        mel_emb_mixed = self.prenet(mixed_mels)
        mel_emb_mixed = self.mel_pos_encoding(mel_emb_mixed)
        dec_out_mixed = self.decoder(
            mel_emb_mixed, memory,
            tgt_mask=tgt_causal,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad
        )

        mel_before = self.mel_linear(dec_out_mixed)
        gate_out = self.gate_linear(dec_out_mixed).squeeze(-1)
        return mel_before, gate_out

    @torch.no_grad()
    def inference(self, text, max_len=None):
        """
        Autoregressive generation, max_len= limit
        text: LongTensor [1, T_text] (pad-free ideally)
        returns: mel [1, T_out, n_mels]
        """
        self.eval()
        device = text.device
        max_len = max_len or self.hp.max_len

        #lowercase upstream
        B = 1
        T_text = text.size(1)
        text_emb = self.text_embedding(text)
        text_emb = self.text_pos_encoding(text_emb)
        src_pad = torch.zeros((B, T_text), device=device, dtype=torch.bool)
        memory = self.encoder(text_emb, src_key_padding_mask=src_pad)

        # SOS mel frame
        mel_outputs = torch.zeros((B, 1, self.hp.n_mels), device=device)

        for step in range(max_len):
            T_mel = mel_outputs.size(1)
            mel_emb = self.prenet(mel_outputs)
            mel_emb = self.mel_pos_encoding(mel_emb)
            tgt_causal = self._causal_mask(T_mel, device)

            dec_out = self.decoder(
                mel_emb, memory,
                tgt_mask=tgt_causal,
                memory_key_padding_mask=src_pad
            )

            last_state = dec_out[:, -1:, :]
            next_mel_before = self.mel_linear(last_state)  # [B, 1, n_mels]
            if self._postnet_enabled:
                next_mel = next_mel_before + self.postnet(next_mel_before)
            else:
                next_mel = next_mel_before

            gate_logit = self.gate_linear(last_state).squeeze(-1).squeeze(-1)  # [B]

            mel_outputs = torch.cat([mel_outputs, next_mel], dim=1)

            if (step + 1) >= self.hp.min_inference_len:
                if torch.sigmoid(gate_logit).item() > self.hp.stop_threshold:
                    break

        # drop SOS frame
        return mel_outputs[:, 1:, :]
