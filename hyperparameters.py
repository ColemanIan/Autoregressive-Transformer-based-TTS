import torch
import string
import os


class Hyperparameters:
    # -----------------------
    # Text tokens / vocab
    # -----------------------
    PAD, SOS, EOS, UNK = 0, 1, 2, 3

    _chars = [' '] + list(string.ascii_lowercase + "'-,.?!")
    vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + _chars
    vocab_size = len(vocab)
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for i, c in enumerate(vocab)}

    def __init__(self):
        # -----------------------
        # System
        # -----------------------
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 42

        # -----------------------
        # Paths
        # -----------------------
        self.data_dir = './data'
        self.output_dir = './output'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f'{self.output_dir}/checkpoints', exist_ok=True)
        os.makedirs(f'{self.output_dir}/plots', exist_ok=True)
        os.makedirs(f'{self.output_dir}/audio', exist_ok=True)

        # -----------------------
        # Audio
        # -----------------------
        self.sample_rate = 22050
        self.n_fft = 2048
        self.hop_length = 256
        self.win_length = 1024
        self.n_mels = 80
        self.n_stft = self.n_fft // 2 + 1
        self.top_db = 80

        # -----------------------
        # Model architecture
        # -----------------------
        # Core transformer dims
        self.d_model = 256           # transformer model width
        self.num_layers = 4          # encoder layers = decoder layers
        self.num_heads = 4           # keep head_dim ~ d_model / num_heads ≈ 64
        self.ff_multiplier = 4       # FFN expansion (d_ff = ff_multiplier * d_model)
        self.d_ff = self.ff_multiplier * self.d_model
        self.dropout = 0.1
        self.max_len = 1000          # for positional encodings

        # Decoder-side modules
        self.prenet_hidden = 256     # hidden width for prenet (mel -> hidden)
        self.prenet_dropout = 0.1    # strong dropout helps AR stability
        self.keep_prenet_dropout_at_inference = True  # set True to mimic Tacotron trick

        # Gate head MLP
        self.gate_hidden = 128
        self.gate_bias_init = -2.0   # pessimistic initial bias for "not stop" (sigmoid≈0.12)

        # -----------------------
        # Training
        # -----------------------
        self.batch_size = 32
        self.num_epochs = 20
        self.lr = 2e-6                 # initial learning rate (will be scheduled)
        self.warmup_steps = 1000       # warmup steps for scheduler
        self.grad_clip = 1.0
        self.gate_pos_weight = 5.0    # positive class weight for BCEWithLogitsLoss
        
        # Progressive scheduled sampling
        self.scheduled_sampling_start = 0.9  # Start with 90% teacher forcing
        self.scheduled_sampling_end = 0.1    # End with 10% teacher forcing
        self.scheduled_sampling_ratio = 0.9  # Current ratio (will be updated during training)

        # -----------------------
        # Inference
        # -----------------------
        self.stop_threshold = 0.5     # sigmoid(gate) > threshold => stop (after min len)
        self.min_inference_len = 20   # don't stop too early


        # -----------------------
        # Postnet
        # -----------------------
        self.postnet_channels = 512
        self.postnet_kernel_size = 5
        self.postnet_dropout = 0.5


    @classmethod
    def text_to_sequence(cls, text):
        """Convert text to token indices"""
        text = str(text).lower()
        seq = [cls.SOS]
        seq.extend(cls.stoi.get(c, cls.UNK) for c in text)
        seq.append(cls.EOS)
        return seq

    @classmethod
    def sequence_to_text(cls, seq):
        """Convert token indices to text"""
        chars = []
        for idx in seq:
            if idx == cls.EOS:
                break
            if idx > cls.EOS:
                chars.append(cls.itos.get(idx, ""))
        return ''.join(chars)


hp = Hyperparameters()