# Kaggle deployment — full training run (free)

Procedure for training on Kaggle's free GPU tier (2× T4 15 GB). Cost: **$0**. Expected wall time: **5-8 hours** for the full run. We use a **single T4** — multi-GPU adds complexity and only ~1.6× speedup due to communication overhead on small models.

> **About bf16:** T4 (Turing, 2018) does not support bf16 natively. The training script checks the GPU's compute capability (Ampere ≥ 8.0) and falls back to fp16 on T4 automatically. No code change needed.

## 🚀 Fastest path: use the prebuilt notebook

The repo ships a ready-to-upload Kaggle notebook at [`kaggle/ryGPT_train.ipynb`](../kaggle/ryGPT_train.ipynb). It has all the cells wired up, with checks for common failure modes (internet off, dataset missing, wrong dataset slug).

**Steps:**

1. Upload your data as a private Kaggle dataset (see section below)
2. Create a new Kaggle notebook → **File → Import Notebook** → upload `kaggle/ryGPT_train.ipynb`
3. In notebook settings: **GPU T4 ×1** + **Internet On** + **Add Data → rygpt-data**
4. Click **Run All**

Skip the manual step-by-step below unless you want to understand each cell.

---

## Manual walkthrough (equivalent to the notebook, for reference)

## 1. Pre-flight (do this locally first)

- [ ] Review `eval/anon_review_samples.txt` for any PII the regex missed (family names, addresses, school/workplace).
- [ ] (Optional) Drop Manglish negative-class snippets into `data/eval_negatives/*.txt` for the style classifier.

## 2. Upload your data as a private Kaggle dataset

Kaggle expects training data as a "dataset" (separate from the notebook). One-time setup:

1. Go to **kaggle.com/datasets** → **+ New Dataset**
2. Drag in these 3 files:
   - `data/processed/train.jsonl` (~166 MB)
   - `data/processed/val.jsonl` (~18 MB)
   - `data/anonymized/name_mapping.json` (~1 KB)
3. **Title:** `rygpt-data` (or any slug)
4. **Visibility:** **Private** (important — your messages are in there)
5. Click **Create**

The dataset gets a URL like `kaggle.com/datasets/<your-username>/rygpt-data`.

## 3. Create the notebook

1. **+ New Notebook**
2. Settings (right sidebar):
   - **Accelerator:** GPU T4 ×1 (or T4 ×2 if you want to experiment with multi-GPU later)
   - **Internet:** **On** (needed to download Qwen2.5 weights from HuggingFace)
   - **Persistence:** Variables and Files (so adapter survives)
3. **Add Data** (right sidebar) → search for your `rygpt-data` dataset → Add. It mounts at `/kaggle/input/rygpt-data/`.

## 4. Notebook cells (copy-paste in order)

### Cell 1 — environment check
```python
!nvidia-smi
import torch
print('CUDA:', torch.cuda.is_available(),
      '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-',
      '| bf16 supported:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
```
Expected: `Tesla T4 | bf16 supported: False`. The training script will auto-fall-back to fp16.

### Cell 2 — clone repo + install deps
```bash
%%bash
cd /kaggle/working
git clone https://github.com/rihaans/ryGPT.git
cd ryGPT
pip install -q -r requirements.txt
pip install -q bitsandbytes
```

### Cell 3 — wire the uploaded dataset into the expected paths
```bash
%%bash
cd /kaggle/working/ryGPT
mkdir -p data/processed data/anonymized
cp /kaggle/input/rygpt-data/train.jsonl       data/processed/train.jsonl
cp /kaggle/input/rygpt-data/val.jsonl         data/processed/val.jsonl
cp /kaggle/input/rygpt-data/name_mapping.json data/anonymized/name_mapping.json
ls -lh data/processed data/anonymized
```

### Cell 4 — kick off training (long-running)
T4-tuned defaults: batch 8 instead of 16 (15 GB VRAM), grad_accum 2 to keep effective batch = 16.
```bash
%%bash
cd /kaggle/working/ryGPT
python scripts/06_train_model.py \
    --base-model Qwen/Qwen2.5-1.5B \
    --batch-size 8 \
    --grad-accum 2 \
    --max-seq-length 256 \
    --epochs 3 \
    --eval-steps 1000 \
    --save-steps 1000 \
    --logging-steps 50 \
    --wandb-disabled \
    --out-dir /kaggle/working/ryGPT/models/lora_adapter
```

**The session times out at 12 hours.** If you set `--epochs 3` and it doesn't finish, use `--epochs 1` first; the auto-saved checkpoints can be resumed by manually pointing to them (or just run again — it's deterministic given the seed).

### Cell 5 — run eval (Phase 7)
```bash
%%bash
cd /kaggle/working/ryGPT
python scripts/07_evaluate.py
```

### Cell 6 — show the eval markdown inline
```python
from pathlib import Path
for name in ('perplexity', 'memorization', 'samples', 'style_classifier'):
    p = Path(f'/kaggle/working/ryGPT/eval/{name}.md')
    if p.exists():
        print(f'==== {name} ====')
        print(p.read_text(encoding='utf-8'))
        print()
```

### Cell 7 — package the adapter into a downloadable archive
```bash
%%bash
cd /kaggle/working/ryGPT
tar -czf /kaggle/working/rygpt_lora_adapter.tar.gz models/lora_adapter eval/*.md
ls -lh /kaggle/working/rygpt_lora_adapter.tar.gz
```

## 5. Download the adapter

In the notebook, the right sidebar **Output** section shows files saved under `/kaggle/working/`. Find `rygpt_lora_adapter.tar.gz` and click the download icon.

On your laptop:
```powershell
cd C:\Users\rihaa\Development\Projects\ryGPT
tar -xzf rygpt_lora_adapter.tar.gz   # extracts models/lora_adapter/ and eval/*.md
```

## 6. Chat with it locally

No GPU needed for inference (the laptop 4070 is plenty):
```powershell
python scripts/chat.py --adapter-dir models/lora_adapter
```

## Kaggle quirks and gotchas

- **30 hours/week** of T4 quota. A full 8-hour training run + 30 min eval uses about 1/3 of your weekly budget.
- **12 hour session limit.** If training looks like it'll exceed this, drop to `--epochs 2` or `--epochs 1`.
- **Notebooks don't auto-save outputs unless you commit.** Click the **Save & Run All** (top-right) when you're ready to keep the run permanently — Kaggle then re-runs the whole notebook headlessly. Or, for a long live session: **Save Version → Quick Save** captures the current state.
- **Idle timeout.** Kaggle kills idle interactive sessions after ~20 min. Either (a) keep the browser tab open and active, or (b) use **Save & Run All** which runs headlessly without an open tab.
- **Internet off by default for some notebooks.** Double-check the **Internet: On** toggle in settings before Cell 2 (HF download will fail otherwise).
- **HF rate limiting.** Recommended: create a HuggingFace account, generate a read token at huggingface.co/settings/tokens, add it as a **Kaggle Secret** (Add-ons → Secrets), then set `os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')` in Cell 1.

## Speed estimate (single T4 vs 4090)

| GPU         | steps/sec (effective batch 16) | full 3-epoch run |
|-------------|-------------------------------:|-----------------:|
| RTX 4090    | ~5                             | ~3 hr            |
| Tesla T4    | ~1.5-2                         | ~6-8 hr          |

T4 is roughly 1/3 the throughput of a 4090 — same model trains, just takes longer.

## When to use Kaggle vs RunPod

| if … | use … |
|---|---|
| Budget = $0 and 5-8 hr wall time is fine | **Kaggle** |
| Want it done in 3 hours and $3 is fine | **RunPod** (see `RUNPOD_DEPLOY.md`) |
| Want to iterate hyperparameters multiple times | RunPod (quota concerns on Kaggle) |
| First-time fine-tuner, want simplest UX | Kaggle (notebook is easier than SSH) |
