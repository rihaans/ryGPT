# RunPod deployment — full training run

End-to-end procedure for the Phase 6 + 7 cloud run. Total est. cost: **$4-6**, total est. time on a 4090: **~3-4 hours**.

## 1. Pre-flight (do this locally, before paying)

- [ ] Manually review **`eval/anon_review_samples.txt`** — scan the 20 sampled targets for any leaked PII (family names, addresses, school/workplace names, etc.). If you find any, add to `data/anonymized/name_mapping.json` as aliases and re-run Phases 2 → 3 → 4 → 6.
- [ ] **(Optional but recommended)** Drop scraped Manglish text into `data/eval_negatives/*.txt`:
  - One line per snippet
  - Sources: r/Kerala posts, public Telegram Manglish channels
  - Without this, the Phase 7 style classifier just gets skipped (no error).
- [ ] Confirm files exist locally: `data/processed/train.jsonl`, `data/processed/val.jsonl`, `data/anonymized/name_mapping.json`.

## 2. Provision the pod

1. Sign up at **runpod.io** (Google login works). Add **$10 credit**.
2. **Deploy** → **GPU Cloud** → search **RTX 4090** → pick **Secure Cloud** (~$0.69/hr).
3. **Pod template:** `PyTorch 2.5 + CUDA 12.1` (or any image with CUDA 12.1 + Python 3.10+).
4. **Volume disk:** 30 GB is plenty (model + data + checkpoints).
5. **Network volume:** skip — we'll download the adapter at the end.
6. Click **Deploy On-Demand**.

When the pod is ready: open the **Web Terminal** (button on the pod row).

## 3. Set up the environment on the pod

```bash
# Clone the repo
git clone https://github.com/rihaans/ryGPT.git
cd ryGPT

# Install Python deps (bitsandbytes is happy on Linux)
pip install -r requirements.txt
pip install bitsandbytes

# Verify CUDA is alive
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Upload your data

The training data is local-only (gitignored). Three options to upload, fastest first:

### Option A — scp from your laptop (fastest)

On your laptop:
```powershell
# RunPod gives you an SSH command on the pod page. Adapt it for scp:
scp -P <pod-ssh-port> -r data\processed root@<pod-ip>:/workspace/ryGPT/data/
scp -P <pod-ssh-port> data\anonymized\name_mapping.json root@<pod-ip>:/workspace/ryGPT/data/anonymized/
```

### Option B — `runpodctl send / receive` (RunPod's built-in transfer)

On the pod: `runpodctl receive` (gives you a code).
On your laptop, after installing runpodctl: `runpodctl send data\processed` and enter the code.

### Option C — JupyterLab upload (slowest, click-driven)

Open JupyterLab from the pod page → drag-and-drop `train.jsonl` and `val.jsonl` into `data/processed/`, and `name_mapping.json` into `data/anonymized/`.

## 5. Kick off training

Recommended command for an RTX 4090 (24 GB):

```bash
python scripts/06_train_model.py \
    --base-model Qwen/Qwen2.5-1.5B \
    --batch-size 16 \
    --grad-accum 1 \
    --max-seq-length 256 \
    --epochs 3 \
    --eval-steps 500 \
    --save-steps 500 \
    --logging-steps 50 \
    --wandb-disabled
```

Expected ~3 hours for 3 full epochs on 300k examples. Watch the eval_loss column — if it plateaus or starts going up after step ~10k, kill the job (early stopping with patience=3 will auto-trigger anyway).

If you want W&B logging instead of `--wandb-disabled`:
```bash
wandb login   # paste API key from wandb.ai
# drop --wandb-disabled from the train command
```

## 6. Run eval (Phase 7)

```bash
python scripts/07_evaluate.py
```

Expected ~30 min. Produces:
- `eval/perplexity.md` — base vs tuned, broken down by relationship
- `eval/samples.md` — 30 generation samples (10 per relationship), base vs tuned
- `eval/style_classifier.md` — only if `data/eval_negatives/*.txt` exists
- `eval/memorization.md` — hard gate on the demo

**Open `eval/memorization.md` first.** If it says `FAILED — do not deploy demo`, the demo won't launch (this is a safety gate). Inspect flagged examples — many will be short common Manglish phrases ("ariyilla", "ok da") that aren't real memorization. If real memorization, retrain with `--epochs 1` or `--lora-dropout 0.1`.

## 7. Download the trained adapter back to your laptop

The adapter folder is **~50 MB** (LoRA-only, no embedding resize per ADR-010).

```powershell
# On your laptop:
scp -P <pod-ssh-port> -r root@<pod-ip>:/workspace/ryGPT/models/lora_adapter models/
scp -P <pod-ssh-port> -r root@<pod-ip>:/workspace/ryGPT/eval/*.md eval/
```

Or use `runpodctl send` from the pod side.

## 8. Stop the pod

**RunPod keeps charging until you stop the pod.** Stop it from the dashboard the moment you're done downloading.

## 9. Local: chat with the trained model

Back on your laptop (no GPU rental needed for inference):
```powershell
python scripts/chat.py --adapter-dir models/lora_adapter
```

Or the Gradio demo:
```powershell
python scripts/08_demo.py
```

## Cost projection (RTX 4090 Secure, $0.69/hr)

| step                       | time   | cost |
|----------------------------|-------:|-----:|
| Setup + data upload        | 10 min | $0.12 |
| Training (3 epochs)        | ~3 hr  | $2.07 |
| Eval (Phase 7)             | 30 min | $0.35 |
| Adapter download + cleanup | 10 min | $0.12 |
| **Total**                  | ~4 hr  | **~$2.66** |

Budget **$5-6** to leave headroom for retries (e.g. you tweak hyperparameters and re-train).

## Common gotchas

- **Pod times out / disconnects mid-training:** RunPod Secure pods don't get preempted, but the SSH session might. Run training inside `tmux` or `screen`:
  ```bash
  tmux new -s train
  # … run the command …
  # Detach with Ctrl-B then D. Reattach later with `tmux attach -t train`.
  ```
- **Out of disk:** if your volume fills with checkpoints, raise `--save-total-limit` lower (currently 2) or clear old `models/lora_adapter/checkpoint-*` folders.
- **CUDA OOM:** drop `--batch-size` from 16 → 8 → 4. With `--max-seq-length 256` and a 4090, OOM is very unlikely.
- **Hugging Face download throttling:** set `HF_TOKEN` env var with a token from huggingface.co/settings/tokens to lift rate limits.
