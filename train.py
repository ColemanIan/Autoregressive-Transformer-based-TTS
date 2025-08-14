import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

from hyperparameters import hp
from model import TransformerTTS
from dataset import prepare_dataset, get_dataloaders


def calculate_scheduled_sampling_ratio(epoch, total_epochs, start_ratio=0.9, end_ratio=0.1):
    progress = (epoch - 1) / max(1, total_epochs - 1)
    ratio = start_ratio - (start_ratio - end_ratio) * progress
    return max(end_ratio, min(start_ratio, ratio))


def get_cosine_with_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)


def masked_loss(loss, lengths):
    B, T = loss.shape[:2]
    device = loss.device
    mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
    if loss.dim() > 2:
        loss = loss.mean(dim=-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def _align_and_mel_l1(pred_bt_f, target_bt_f, target_lens):
    """
    Compute L1 mel loss between prediction and target with temporal alignment:
    For each item, truncate both to min(pred_T, target_len) and average over time and channels.
    """
    B = target_bt_f.size(0)
    device = target_bt_f.device
    loss_sum = torch.tensor(0.0, device=device)
    denom = 0
    for b in range(B):
        T_tar = int(target_lens[b].item())
        T_pred = int(pred_bt_f[b].size(0))
        T = max(1, min(T_tar, T_pred))
        loss_sum = loss_sum + F.l1_loss(pred_bt_f[b, :T], target_bt_f[b, :T], reduction='mean')
        denom += 1
    return loss_sum / max(1, denom)


def train_epoch(model, loader, optimizer, scheduler, device, gate_weight=100.0, epoch=1):
    model.train()
    total_loss = 0
    total_mel_loss = 0
    total_gate_loss = 0

    pbar = tqdm(loader, desc=f'Training Epoch {epoch}')
    for batch_idx, (seq, seq_lens, mels, mel_lens, gates) in enumerate(pbar):
        current_lr = optimizer.param_groups[0]['lr']

        seq = seq.to(device)
        seq_lens = seq_lens.to(device)
        mels = mels.to(device)
        mel_lens = mel_lens.to(device)
        gates = gates.to(device)

        optimizer.zero_grad()
        mel_out, gate_out = model(seq, seq_lens, mels, mel_lens, use_scheduled_sampling=True)

        mel_loss = F.l1_loss(mel_out, mels, reduction='none')
        mel_loss = masked_loss(mel_loss, mel_lens)

        gate_loss = F.binary_cross_entropy_with_logits(
            gate_out, gates,
            pos_weight=torch.tensor(gate_weight, device=device),
            reduction='none'
        )
        gate_loss = masked_loss(gate_loss, mel_lens)

        loss = mel_loss + gate_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
        optimizer.step()

        if scheduler is not None and hasattr(scheduler, 'step'):
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

        total_loss += loss.item()
        total_mel_loss += mel_loss.item()
        total_gate_loss += gate_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'mel': f'{mel_loss.item():.3f}',
            'gate': f'{gate_loss.item():.3f}',
            'lr': f'{current_lr:.2e}'
        })

    n = len(loader)
    return total_loss / n, total_mel_loss / n, total_gate_loss / n


@torch.no_grad()
def validate_teacher_forcing(model, loader, device, gate_weight=100.0):
    """Teacher-forcing validation (masked mel + gate)."""
    model.eval()
    total_loss = 0
    total_mel_loss = 0
    total_gate_loss = 0

    for seq, seq_lens, mels, mel_lens, gates in tqdm(loader, desc='Validation (Teacher Forcing)'):
        seq = seq.to(device)
        seq_lens = seq_lens.to(device)
        mels = mels.to(device)
        mel_lens = mel_lens.to(device)
        gates = gates.to(device)

        mel_out, gate_out = model(seq, seq_lens, mels, mel_lens, use_scheduled_sampling=False)

        mel_loss = F.l1_loss(mel_out, mels, reduction='none')
        mel_loss = masked_loss(mel_loss, mel_lens)

        gate_loss = F.binary_cross_entropy_with_logits(
            gate_out, gates,
            pos_weight=torch.tensor(gate_weight, device=device),
            reduction='none'
        )
        gate_loss = masked_loss(gate_loss, mel_lens)

        loss = mel_loss + gate_loss
        total_loss += loss.item()
        total_mel_loss += mel_loss.item()
        total_gate_loss += gate_loss.item()

    n = len(loader)
    return total_loss / n, total_mel_loss / n, total_gate_loss / n


@torch.no_grad()
def batched_autoregressive_inference(
    model,
    text_batch,          # LongTensor [B, T_text]
    text_lens,           # LongTensor [B]
    max_len_cap,         # int (global cap)
    use_postnet=True,    # match real inference behavior
):
    """
    Batched autoregressive decode for the whole batch at once.
    Recomputes the full target prefix each step (vanilla nn.Transformer),
    but avoids Python-level per-item loops. Early-stops per item via gate.

    Returns:
        mel_pred: FloatTensor [B, T_out_max, n_mels]
        pred_lens: LongTensor [B] (per-sample decoded lengths)
    """
    device = text_batch.device
    B = text_batch.size(0)
    hp = model.hp

    # Encode text once for the whole batch
    model.eval()
    prev_postnet = model._postnet_enabled
    model.set_postnet(use_postnet)

    # Text encoding
    text_emb = model.text_embedding(text_batch)
    text_emb = model.text_pos_encoding(text_emb)
    src_pad = model._build_key_padding_mask(text_lens, text_batch.size(1), device)
    memory = model.encoder(text_emb, src_key_padding_mask=src_pad)

    # Init mel with a single zero SOS frame
    mel_outputs = torch.zeros((B, 1, hp.n_mels), device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    pred_lens = torch.zeros(B, dtype=torch.long, device=device)

    # Decode loop (batched)
    for step in range(max_len_cap):
        T_mel = mel_outputs.size(1)
        tgt_causal = model._causal_mask(T_mel, device)

        # Prenet + posenc on the whole prefix
        mel_emb = model.prenet(mel_outputs)
        mel_emb = model.mel_pos_encoding(mel_emb)

        # Decode and produce next frame from the last state
        dec_out = model.decoder(
            mel_emb, memory,
            tgt_mask=tgt_causal,
            memory_key_padding_mask=src_pad
        )
        last_state = dec_out[:, -1:, :]                       # [B, 1, D]
        next_mel_before = model.mel_linear(last_state)        # [B, 1, n_mels]
        next_mel = (next_mel_before + model.postnet(next_mel_before)) if model._postnet_enabled else next_mel_before
        gate_logit = model.gate_linear(last_state).squeeze(-1).squeeze(-1)  # [B]

        # If an item just finished at this step, set its predicted length
        newly_finished = (torch.sigmoid(gate_logit) > hp.stop_threshold) & (~finished) & ((step + 1) >= hp.min_inference_len)
        pred_lens = torch.where(newly_finished, torch.tensor(step + 1, device=device), pred_lens)
        finished = finished | newly_finished

        # For items already finished, keep their last frame (no change)
        if finished.any():
            keep_prev = finished.view(B, 1, 1)
            next_mel = torch.where(keep_prev, mel_outputs[:, -1:, :], next_mel)

        mel_outputs = torch.cat([mel_outputs, next_mel], dim=1)

        # If all finished, break
        if finished.all():
            break

    # If any never finished, they ran to max_len_cap; set their length accordingly
    pred_lens = torch.where(pred_lens == 0, torch.tensor(mel_outputs.size(1) - 1, device=device), pred_lens)

    # Drop the SOS frame
    mel_outputs = mel_outputs[:, 1:, :]  # [B, T_out, n_mels]

    # Restore original PostNet setting
    model.set_postnet(prev_postnet)
    return mel_outputs, pred_lens



@torch.no_grad()
def validate_inference(model, loader, device, length_margin=16, use_postnet=True, use_amp=True):
    """
    Inference-based validation using the batched decoder above.
    - Caps per-sample max_len to (target_len - 1 + margin)
    - Runs batches together to reduce Python overhead
    - Computes per-item mel L1 on aligned lengths and averages across the set
    """
    model.eval()
    total_mel_loss = 0.0
    n_batches = 0

    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if (use_amp and device.type == 'cuda') else torch.cuda.amp.autocast(enabled=False)

    with torch.inference_mode(), amp_ctx:
        for seq, seq_lens, mels, mel_lens, _gates in tqdm(loader, desc='Validation (Inference, Batched)'):
            seq = seq.to(device)
            seq_lens = seq_lens.to(device)
            mels = mels.to(device)
            mel_lens = mel_lens.to(device)

            # Per-batch cap: use the max target len in the batch + margin (and clip to hp.max_len)
            tgt_Ts = (mel_lens - 1).clamp_min(1)
            cap = int(torch.minimum(tgt_Ts.max() + length_margin, torch.tensor(model.hp.max_len, device=device)).item())

            # Batched AR inference
            pred_mels, pred_lens = batched_autoregressive_inference(
                model, seq, seq_lens, max_len_cap=cap, use_postnet=use_postnet
            )  # pred_mels: [B, T_pred_max, n_mels]; pred_lens: [B]

            # Remove SOS from targets
            targets = mels[:, 1:, :]
            target_lens = mel_lens - 1

            # Compute aligned L1 mel loss per item
            B = seq.size(0)
            loss_sum = 0.0
            for b in range(B):
                T_pred = int(pred_lens[b].item())
                T_tar = int(target_lens[b].item())
                T = max(1, min(T_pred, T_tar))
                loss_sum += F.l1_loss(pred_mels[b, :T], targets[b, :T], reduction='mean').item()

            total_mel_loss += loss_sum / B
            n_batches += 1

    return total_mel_loss / max(1, n_batches)


def visualize_training(epoch, model, val_loader, device, save_path, current_lr, current_ss_ratio):
    """
    Plot GT, Teacher-Forcing prediction, and Inference prediction for the first item in a val batch.
    Also include teacher-forcing percentage in the figure title.
    """
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        seq, seq_lens, mels, mel_lens, gates = [t.to(device) for t in batch]

        # First item only for visualization
        seq = seq[:1]
        seq_lens = seq_lens[:1]
        mels = mels[:1]
        mel_lens = mel_lens[:1]

        # Teacher-forcing prediction
        mel_pred_tf, _gate_pred_tf = model(seq, seq_lens, mels, mel_lens, use_scheduled_sampling=False)

        # Inference prediction
        mel_pred_inf = model.inference(seq[:, :seq_lens[0]])  # [1, T_inf, n_mels]

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        L = int(mel_lens[0].item())
        gt = mels[0, :L].T.cpu()              # includes SOS at t=0
        tf = mel_pred_tf[0, :L].T.cpu()
        inf = mel_pred_inf[0].T.cpu()

        axes[0].imshow(gt, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Ground Truth')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Mel Channel')

        axes[1].imshow(tf, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title(f'Teacher Forcing ({current_ss_ratio*100:.1f}%)')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Mel Channel')

        axes[2].imshow(inf, aspect='auto', origin='lower', cmap='viridis')
        axes[2].set_title('Inference')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Mel Channel')

        plt.suptitle(f'Epoch {epoch} | LR: {current_lr:.2e}')
        plt.tight_layout()

        plot_path = f'{save_path}/epoch_{epoch:03d}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization to {plot_path}")


def _build_optimizer_and_scheduler(model, learning_rate, total_steps, warmup_steps, scheduler_type, postnet_lr_scale=2.0):
    """
    Optimizer with two parameter groups:
      - base (encoder/decoder/heads)
      - postnet (trained in stage 2) with a scaled LR
    """
    base_params = model.base_parameters()
    postnet_params = model.postnet_parameters()

    param_groups = [
        {"params": base_params, "lr": learning_rate, "weight_decay": 0.01},
        {"params": postnet_params, "lr": learning_rate * postnet_lr_scale, "weight_decay": 0.01},
    ]

    optimizer = AdamW(
        param_groups,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    scheduler = None
    if scheduler_type == 'cosine_warmup':
        scheduler = get_cosine_with_warmup_scheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        print(f"Using Cosine Annealing with {warmup_steps} warmup steps")
    elif scheduler_type == 'warmup':
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f"Using Linear Warmup for {warmup_steps} steps")
    elif scheduler_type == 'cosine_restarts':
        steps_per_restart = max(1, total_steps // 4)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps_per_restart,
            T_mult=1,
            eta_min=learning_rate * 0.01
        )
        print("Using Cosine Annealing with Warm Restarts")
    else:
        print("No learning rate scheduling")

    return optimizer, scheduler


def train(
    save_name="tts_model",
    num_epochs=None,
    learning_rate=None,
    batch_size=None,
    gate_weight=None,
    checkpoint_path=None,
    visualize_every=2,
    use_progressive_sampling=True,
    ss_start=None,
    ss_end=None,
    warmup_steps=1000,
    scheduler_type='cosine_warmup',
    val_mode='inference_and_teacher',
    stage1_epochs=None,
    stage2_epochs=None,
    postnet_lr_scale=2.0,
    df=None
):
    """
    Two-stage training with dual validation:
      - Teacher-forcing validation (mel + gate masked losses)
      - Inference validation (mel loss only, aligned)
    Plots include GT, TF, and Inference spectrograms with TF percent in the title.
    """
    print(f"\nTransformer TTS Training - {save_name}\n")

    # Defaults from hyperparameters
    num_epochs = num_epochs or hp.num_epochs
    learning_rate = learning_rate or hp.lr
    batch_size = batch_size or hp.batch_size
    gate_weight = gate_weight or hp.gate_pos_weight
    ss_start = ss_start or hp.scheduled_sampling_start
    ss_end = ss_end or hp.scheduled_sampling_end

    # Split epochs if not provided
    if stage1_epochs is None and stage2_epochs is None:
        stage1_epochs = max(1, num_epochs // 2)
        stage2_epochs = num_epochs - stage1_epochs
    elif stage1_epochs is None:
        stage1_epochs = max(0, num_epochs - stage2_epochs)
    elif stage2_epochs is None:
        stage2_epochs = max(0, num_epochs - stage1_epochs)

    # Paths
    checkpoint_dir = f'{hp.output_dir}/checkpoints/{save_name}'
    plot_dir = f'{hp.output_dir}/plots/{save_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Device and data
    device = torch.device(hp.device)
    torch.manual_seed(hp.seed)
    if df is None:
        df = prepare_dataset()
    train_loader, val_loader = get_dataloaders(df, batch_size)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * (stage1_epochs + stage2_epochs)

    # Model
    model = TransformerTTS(hp).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")

    # Optimizer/scheduler
    optimizer, scheduler = _build_optimizer_and_scheduler(
        model, learning_rate, total_steps, warmup_steps, scheduler_type, postnet_lr_scale
    )

    best_tf_loss = float('inf')
    best_inf_mel_loss = float('inf')

    history = {
        'train_loss': [], 'train_mel': [], 'train_gate': [],
        'val_tf_loss': [], 'val_tf_mel': [], 'val_tf_gate': [],
        'val_inf_mel': [],
        'learning_rates': [], 'scheduled_sampling_ratios': []
    }

    def _do_epoch_range(first_epoch, last_epoch, stage_name, postnet_enabled):
        nonlocal best_tf_loss, best_inf_mel_loss
        model.set_postnet(postnet_enabled)
        print(f"\n{stage_name} | epochs {first_epoch} to {last_epoch} | postnet_enabled={postnet_enabled}")

        total_epochs = stage1_epochs + stage2_epochs

        for epoch in range(first_epoch, last_epoch + 1):
            # Teacher forcing ratio (current_ss_ratio)
            if use_progressive_sampling:
                current_ss_ratio = calculate_scheduled_sampling_ratio(
                    epoch, total_epochs, ss_start, ss_end
                )
                hp.scheduled_sampling_ratio = current_ss_ratio
                print(f"\nEpoch {epoch}/{total_epochs} | Teacher Forcing: {current_ss_ratio*100:.1f}%")
            else:
                current_ss_ratio = hp.scheduled_sampling_ratio
                print(f"\nEpoch {epoch}/{total_epochs}")

            # Train
            train_loss, train_mel, train_gate = train_epoch(
                model, train_loader, optimizer, scheduler, device, gate_weight, epoch
            )
            history['train_loss'].append(train_loss)
            history['train_mel'].append(train_mel)
            history['train_gate'].append(train_gate)

            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            history['scheduled_sampling_ratios'].append(current_ss_ratio)

            # Validate (both TF and Inference)
            tf_loss, tf_mel, tf_gate = validate_teacher_forcing(model, val_loader, device, gate_weight)
            inf_mel = validate_inference(model, val_loader, device)

            history['val_tf_loss'].append(tf_loss)
            history['val_tf_mel'].append(tf_mel)
            history['val_tf_gate'].append(tf_gate)
            history['val_inf_mel'].append(inf_mel)

            print(f"Train Loss: {train_loss:.4f} (Mel: {train_mel:.4f}, Gate: {train_gate:.4f})")
            print(f"Val TF   : {tf_loss:.4f} (Mel: {tf_mel:.4f}, Gate: {tf_gate:.4f})")
            print(f"Val INF  : Mel: {inf_mel:.4f}")

            # Checkpoints: save latest; keep best by TF loss and by INF mel loss
            checkpoint_common = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'history': history,
            }
            torch.save({**checkpoint_common, 'metric': tf_loss}, f'{checkpoint_dir}/latest.pt')

            if tf_loss < best_tf_loss:
                best_tf_loss = tf_loss
                torch.save({**checkpoint_common, 'metric': tf_loss}, f'{checkpoint_dir}/best_tf.pt')
                print(f"Saved best TF model (tf_loss: {best_tf_loss:.4f})")

            if inf_mel < best_inf_mel_loss:
                best_inf_mel_loss = inf_mel
                torch.save({**checkpoint_common, 'metric': inf_mel}, f'{checkpoint_dir}/best_inf.pt')
                print(f"Saved best INF model (inf_mel: {best_inf_mel_loss:.4f})")

            # Visualize GT / TF / INF
            if epoch % max(1, visualize_every) == 0:
                visualize_training(epoch, model, val_loader, device, plot_dir, current_lr, current_ss_ratio)

    # Stage 1: no PostNet
    if stage1_epochs > 0:
        _do_epoch_range(1, stage1_epochs, "Stage 1 (no postnet)", postnet_enabled=False)

    # Stage 2: enable PostNet
    if stage2_epochs > 0:
        _do_epoch_range(stage1_epochs + 1, stage1_epochs + stage2_epochs, "Stage 2 (with postnet)", postnet_enabled=True)

    # Summary plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes[0, 0].plot(history['train_loss'], label='Train Total')
    axes[0, 0].plot(history['val_tf_loss'], label='Val TF Total')
    axes[0, 0].set_title('Total Loss (Train vs Val TF)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['train_mel'], label='Train Mel')
    axes[0, 1].plot(history['val_tf_mel'], label='Val TF Mel')
    axes[0, 1].plot(history['val_inf_mel'], label='Val INF Mel')
    axes[0, 1].set_title('Mel Loss (Train / TF / INF)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mel L1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history['train_gate'], label='Train Gate')
    axes[0, 2].plot(history['val_tf_gate'], label='Val TF Gate')
    axes[0, 2].set_title('Gate Loss (Train / TF)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('BCE')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(history['learning_rates'], label='Learning Rate')
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Update')
    axes[1, 0].set_ylabel('LR')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(history['scheduled_sampling_ratios'], label='Teacher Forcing Ratio')
    axes[1, 1].set_title('Teacher Forcing Ratio')
    axes[1, 1].set_xlabel('Update')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # Generalization gap between Train total and Val TF total
    gap = np.array(history['val_tf_loss']) - np.array(history['train_loss'])
    axes[1, 2].plot(gap, label='ValTF - Train')
    axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 2].set_title('Generalization Gap (TF)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss Gap')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/training_history.png', dpi=110, bbox_inches='tight')
    plt.close()

    print(f"\nTraining complete.")
    print(f"Best TF loss:  {min(history['val_tf_loss']) if history['val_tf_loss'] else float('inf'):.4f}")
    print(f"Best INF mel:  {min(history['val_inf_mel']) if history['val_inf_mel'] else float('inf'):.4f}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Plots: {plot_dir}")

    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Two-Stage TTS Training (TF + Inference Validation)')
    parser.add_argument('--name', type=str, default='tts_model')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--gate_weight', type=float)
    parser.add_argument('--progressive_sampling', type=bool, default=True)
    parser.add_argument('--ss_start', type=float, default=0.9)
    parser.add_argument('--ss_end', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--scheduler', type=str, default='cosine_warmup',
                        choices=['cosine_warmup', 'warmup', 'cosine_restarts', 'none'])
    parser.add_argument('--val_mode', type=str, default='inference_and_teacher',
                        choices=['inference_and_teacher'])
    parser.add_argument('--stage1_epochs', type=int)
    parser.add_argument('--stage2_epochs', type=int)
    parser.add_argument('--postnet_lr_scale', type=float, default=2.0)

    args = parser.parse_args()

    model, history = train(
        save_name=args.name,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gate_weight=args.gate_weight,
        use_progressive_sampling=args.progressive_sampling,
        ss_start=args.ss_start,
        ss_end=args.ss_end,
        warmup_steps=args.warmup_steps,
        scheduler_type=args.scheduler,
        val_mode=args.val_mode,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        postnet_lr_scale=args.postnet_lr_scale
    )
