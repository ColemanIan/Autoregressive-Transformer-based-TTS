# Transformer TTS — Training Notes and Rationale

## Summary
Training this autoregressive Transformer TTS has proven difficult to stabilize. After extensive debugging of data shapes, masking, tokenization, and optimizer settings, the dominant failure mode appears to be **exposure bias**: the model performs well under teacher forcing but degrades when run autoregressively at inference. To mitigate this, the project now uses **two-stage training**, **scheduled sampling**, and a validation protocol that reports both **teacher-forcing** and **inference** losses each epoch.

This README explains the problem symptoms, the exposure-bias hypothesis, the two-stage solution, and the reasoning behind the optimizer and scheduler choices.

---

## Symptoms Observed
- A consistent gap between **teacher-forcing validation loss** and **inference validation loss**. Predictions during teacher forcing look reasonable, but free-run inference drifts or collapses.
- Stop/gate behavior triggers too late or too early during inference, despite learning under teacher forcing.
- PostNet sometimes hides decoder errors in training, producing good-looking mel spectrograms in TF but brittle behavior in inference.
- Improvements made to data preprocessing, masks, and shapes do not close the TF vs inference gap fully.

These are characteristic signs that the model does not learn to be robust to its **own** past prediction errors, only to the ground-truth inputs it sees during teacher forcing. I also had been including capital letters in my vocabulary, which possibly destabilized the model during inference because there was not enough training data for capital letters. I switched to lowercase letters only and there seems to be some improvement. 

---

## Exposure Bias 
**Exposure bias** arises when a sequence model is trained mostly with teacher forcing: at time step *t*, the decoder conditions on the **ground-truth** frame from time *t-1*. At inference, those ground-truth frames are not available; the model must condition on its **own** previous predictions. If those predictions are slightly off, the errors can propogate until the model's predictions become unstable.

**Mitigations used here:**
1. **Scheduled sampling:** progressively reduce the teacher-forcing ratio, replacing some ground-truth frames with the model’s own predictions during training.
2. **Inference-based validation:** measure a mel loss under real free-run conditions to track whether improvements in TF loss actually translate to inference.
3. **Two-stage training with PostNet late:** stabilize the core AR mapping first, then allow PostNet to refine spectra after the decoder is competent.

---

## Two-Stage Training
**Goal:** Help the decoder learn a stable text-to-mel mapping before adding a refinement network that can mask decoder errors.

### Stage 1: Decoder without PostNet
- **PostNet disabled.**
- The postnet module is completely skipped, so the loss is calculated directly from the decoder output.
- Train encoder, decoder, mel head, and gate head only.
- Use scheduled sampling and teacher-forcing loss to stabilize alignment and timing.
- Rationale: make the decoder truly carry its share of the work; do not let PostNet compensate for systematic decoder errors too early.

### Stage 2: Enable PostNet and continue
- **PostNet enabled**, optionally with a slightly **higher learning rate** than the base network to let it catch up.
- Continue with the same losses and scheduled sampling.
- Rationale: once the decoder is stable, PostNet can safely refine spectral details without hiding alignment or stability issues.

### Validation and Plots 
- Compute **teacher-forcing validation** (masked mel L1 + gate BCE).
- Compute **inference validation** by running the model autoregressively, then aligning lengths and computing mel L1 only.
- Plot three panels from a validation sample:
  1. **Ground Truth** mel
  2. **Teacher Forcing** mel (title shows current teacher-forcing percentage)
  3. **Inference** mel
- Save separate checkpoints: `best_tf.pt` (lowest teacher forcing total loss) and `best_inf.pt` (lowest inference mel loss). When in doubt, save everything. 

---

## Scheduled Sampling
- The training loop uses a **teacher-forcing ratio** that decays from a higher value (0.9) to a lower value (0.1) across epochs.
- At time step *t*, with probability `(1 - ratio)`, the previous **model prediction** is fed to the decoder; otherwise the **ground-truth** frame is used. **LINGO:** one column in the spectrogram = one frame. The scheduled sampling is a mix of true frames and predicted frames. 
- This gradually exposes the model to its own mistakes during training, reducing the teacher-forcing/inference gap.


---

## Optimizer Choice: AdamW
- **AdamW** is used with weight decay. Adam’s per-parameter adaptive learning rates (via first/second moment estimates) are well-suited for the heterogeneous scales in Transformer layers.
- Decoupled weight decay is preferable to L2 in Adam because it behaves more like true weight decay and less like gradient scaling.
- Typical settings used here:
  - `betas=(0.9, 0.999)`
  - `eps=1e-8`
  - Moderate `weight_decay` on most parameters
  - **Gradient clipping** to control exploding gradients in the decoder; particularly helpful for stabilizing the stop_token prediction.
- The **gate** uses a positive class weight (pos_weight) in BCE to encourage decisive stop predictions when appropriate. Anything from 1-5 as a positive weight seems to encourage the stop_token predictions to converge slightly faster or more confident. I was having trouble reaching the threshold value at 0.5.


---

## Scheduler Choice: Cosine Annealing with Warmup
The default schedule is **cosine annealing with a warmup** phase.

- **Warmup:** the learning rate increases linearly from 0 to the base LR over a set number of steps. This helps stabilize the early stages when gradients can be particularly noisy and magnitudes are not yet calibrated.
- **Cosine annealing:** after warmup, the learning rate follows a smooth cosine decay.
- Optionally, **cosine with warm restarts** can be used, which periodically increases the LR to escape local minima.
  
**Other options available in code:**
- `warmup` only: linear warmup followed by a flat LR.
- `cosine_restarts`: cosine cycles with restarts.
- `none`: hold LR constant (simplest, but you may need to lower it and train longer).

## Conclusion
The core challenge has been exposure bias: strong teacher-forced training does not automatically translate into robust autoregressive inference. The two-stage regimen, scheduled sampling, and inference-aware validation directly target this gap. AdamW and cosine annealing with warmup provide stable learning rate adjustments model while keeping the codebase simple.
