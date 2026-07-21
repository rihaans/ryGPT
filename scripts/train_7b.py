"""Phase 6 (7B) — QLoRA fine-tuning of Qwen2.5-7B-Instruct on the Manglish corpus.

Self-contained trainer for the larger model, with robust cross-session
checkpoint/resume (Kaggle's 12h cap means a 7B run spans several sessions).

WHY Qwen2.5-7B-*Instruct* (not the base model used in 06_train_model.py):
  1. The base Qwen2.5 checkpoint never learned that conversation turns *end* —
     its only prior is "text keeps going." A rank-16 LoRA over 2 epochs could
     not override that, so generations ran on forever and drifted into
     foreign-script noise. See .docs/DECISIONS.md ADR-011.
  2. The Instruct variant is chat-tuned: its generation_config.json already
     lists `<|im_end|>` (151645) as a stop token AND it was trained to emit
     `<|im_end|>` after every turn. So it stops on its own — the entire class
     of "never stops" bugs is fixed at the source, not patched at inference.
  3. 7B has materially better contextual coherence than 1.5B — the actual
     lever for reply quality, which is this project's remaining weak point.

The training-example format is IDENTICAL to 06_train_model.py, so the same
data/processed/{train,val}.jsonl works unchanged. The relationship token
(<gf>/<friend>/<group>) rides in the chat template's system slot.

Run on a GPU box (bitsandbytes is Linux+CUDA-only). For Kaggle T4x2:
    accelerate launch --multi_gpu --num_processes 2 scripts/train_7b.py

Reads:  data/processed/{train,val}.jsonl
Writes: models/lora_adapter_7b/  (adapter + tokenizer + training_config.json)
        models/lora_adapter_7b/checkpoint-<step>/  (resume points)
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os
from pathlib import Path

# See 06_train_model.py for the full explanation: loading a 4-bit model with a
# per-process single-device map makes accelerate mistake it for device_map="auto"
# and refuse DDP. This is accelerate's documented escape hatch. Must be set
# before Trainer builds its Accelerator.
os.environ.setdefault("ACCELERATE_BYPASS_DEVICE_MAP", "true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/lora_adapter_7b"))
    # p99 of real targets is 170 tokens; 256 clips only ~0.16% and halves the
    # padding budget vs 512. Same as the 1.5B run — data hasn't changed.
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--max-train-examples", type=int, default=0,
                        help="Cap on training examples (0 = all). Small value for a smoke test.")
    parser.add_argument("--max-val-examples", type=int, default=0)

    # LoRA — same rank/alpha proven on the 1.5B run.
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Optimization. Defaults tuned for a single 16GB T4 running 7B in 4-bit:
    #   batch 2 x grad-accum 8 x 2 GPUs = effective batch 32 (matches 1.5B run).
    # If you OOM, drop to --batch-size 1 --grad-accum 16 (same effective batch).
    parser.add_argument("--lr", type=float, default=1e-4)  # lower than 1.5B's 2e-4: bigger model, already instruct-tuned
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    # epochs=1: the eval_loss on the 1.5B run bottomed at epoch ~1.5 and rose
    # after (overfitting). A 7B instruct model needs even less to adapt *style*
    # (it already knows language + turn structure), and 1 epoch roughly halves
    # the multi-session wall-clock. Bump to 2 only if val loss is still falling.
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Eval / save / log. save-steps == eval-steps (required by
    # load_best_model_at_end). 1000 keeps the "lost work on a crash" window
    # small (~1000 steps) on a run that spans days.
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="ryGPT-7b")
    parser.add_argument("--wandb-disabled", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from accelerate import PartialState
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as e:
        raise SystemExit(
            f"Missing GPU training deps ({e.name}). On the training box run:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "  pip install -r requirements.txt\n"
            "  pip install bitsandbytes\n"
        )

    from src.dataset import example_to_chat_messages, read_jsonl

    is_main = PartialState().is_main_process

    def log(msg: str) -> None:
        if is_main:
            print(msg, flush=True)

    # ---- Tokenizer ----
    # We do NOT add project tokens (<self>/<gf>/<person_N>/[media]) as atomic
    # tokens (ADR-010): the base BPE fragments them into a few sub-pieces, which
    # is fine, and adding new rows to the embedding/lm_head would leave them
    # untrained under a LoRA that only touches attention/MLP.
    log(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Precision: bf16 on Ampere+ (>=8.0), fp16 on Turing (T4 == 7.5) ----
    if torch.cuda.is_available():
        cap_major, _ = torch.cuda.get_device_capability(0)
        use_bf16 = cap_major >= 8
        gpu_name = torch.cuda.get_device_name(0)
    else:
        raise SystemExit("No CUDA GPU detected — 7B QLoRA training needs a GPU.")
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    log(f"GPU: {gpu_name}; precision: {'bf16' if use_bf16 else 'fp16'}")

    # ---- Model (4-bit QLoRA) ----
    log(f"Loading base model in 4-bit: {args.base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    # Per-process single-device map for real data-parallel DDP (a full model
    # copy per GPU), NOT device_map="auto" (which shards layers and runs one
    # GPU at a time). See 06_train_model.py.
    device_map = {"": PartialState().process_index}
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    if is_main:
        model.print_trainable_parameters()

    # ---- Data ----
    log(f"Loading training data from {args.train_file}")
    train_examples = read_jsonl(args.train_file)
    val_examples = read_jsonl(args.val_file)
    log(f"  train={len(train_examples):,}  val={len(val_examples):,}")
    if 0 < args.max_train_examples < len(train_examples):
        import random
        train_examples = random.Random(args.seed).sample(train_examples, args.max_train_examples)
        log(f"  --max-train-examples: subsampled to {len(train_examples):,}")
    if 0 < args.max_val_examples < len(val_examples):
        import random
        val_examples = random.Random(args.seed).sample(val_examples, args.max_val_examples)
        log(f"  --max-val-examples: subsampled to {len(val_examples):,}")

    def tokenize_one(example: dict) -> dict:
        messages = example_to_chat_messages(example)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        full_ids = tokenizer(
            full_text, truncation=True, max_length=args.max_seq_length,
            add_special_tokens=False,
        )["input_ids"]

        # Prefix = everything except the final assistant turn, plus the
        # generation prompt. Loss is computed on the target turn ONLY.
        prefix_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )
        prefix_ids = tokenizer(
            prefix_text, truncation=True, max_length=args.max_seq_length,
            add_special_tokens=False,
        )["input_ids"]

        labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
        if len(labels) != len(full_ids):
            labels = [-100] * len(full_ids)  # truncation broke alignment; drop below
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }

    train_cols = Dataset.from_list(train_examples).column_names
    val_cols = Dataset.from_list(val_examples).column_names
    train_ds = Dataset.from_list(train_examples).map(
        tokenize_one, remove_columns=train_cols, desc="Tokenizing train",
    )
    val_ds = Dataset.from_list(val_examples).map(
        tokenize_one, remove_columns=val_cols, desc="Tokenizing val",
    )
    train_ds = train_ds.filter(lambda r: any(l != -100 for l in r["labels"]))
    val_ds = val_ds.filter(lambda r: any(l != -100 for l in r["labels"]))
    log(f"  after filter: train={len(train_ds):,}  val={len(val_ds):,}")

    # ---- Trainer ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_to = "wandb" if not args.wandb_disabled else "none"

    # load_best_model_at_end requires save_steps to be a multiple of eval_steps.
    # To allow "eval rarely, checkpoint often" (e.g. eval 3000 / save 1000 —
    # fewer expensive eval passes, but a tighter crash-safety net on a
    # multi-session run), auto-disable best-model reload when the pair doesn't
    # satisfy that constraint. At 1 epoch there's little mid-run overfitting,
    # so the final checkpoint is what you want anyway.
    load_best = args.save_steps % args.eval_steps == 0
    if not load_best:
        log(f"  eval-steps={args.eval_steps}, save-steps={args.save_steps}: "
            "save is not a multiple of eval, so load_best_model_at_end is OFF "
            "(final = last checkpoint, not lowest-eval-loss). This is fine for "
            "a 1-epoch run.")

    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        gradient_checkpointing=True,  # essential for 7B to fit on a T4
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        report_to=report_to,
        run_name=f"qlora-{args.base_model.split('/')[-1]}",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding="longest",
        label_pad_token_id=-100, return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    # Persist config for the eval phase to read back (base model, hyperparams).
    if is_main:
        with (args.out_dir / "training_config.json").open("w", encoding="utf-8") as f:
            json.dump({
                "base_model": args.base_model,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "epochs": args.epochs,
                "max_seq_length": args.max_seq_length,
                "seed": args.seed,
            }, f, indent=2)

    # ---- Resume (cross-session) ----
    last_checkpoint = get_last_checkpoint(str(args.out_dir)) if args.out_dir.exists() else None
    total_steps = int(
        len(train_ds) / (args.batch_size * args.grad_accum * PartialState().num_processes)
        * args.epochs
    )
    if last_checkpoint:
        done = int(Path(last_checkpoint).name.split("checkpoint-")[-1])
        log(f"\n=== RESUMING from {last_checkpoint} ===")
        log(f"    step {done} of ~{total_steps} "
            f"(~{100 * done / max(total_steps, 1):.0f}% done) — "
            f"~{total_steps - done} steps left this run\n")
    else:
        log(f"\n=== STARTING FRESH — ~{total_steps} total steps for {args.epochs} epoch(s) ===\n")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # ---- Save final (best) adapter + tokenizer ----
    trainer.save_model(str(args.out_dir))
    if is_main:
        tokenizer.save_pretrained(args.out_dir)
        log(f"\nSaved adapter + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
