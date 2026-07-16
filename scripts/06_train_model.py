"""Phase 6 — QLoRA fine-tuning of a small base model on the Manglish corpus.

Run this on a rented GPU box (RunPod / Lambda / Vast.ai) — bitsandbytes is
Linux+CUDA-only. Local CPU/Windows is not supported.

Setup on the GPU box:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    pip install bitsandbytes

Reads:  data/processed/{train,val}.jsonl, data/anonymized/name_mapping.json
Writes: models/lora_adapter/ (HF format), models/lora_adapter/training_config.json

Run:
    python scripts/06_train_model.py --base-model Qwen/Qwen2.5-1.5B
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os
from pathlib import Path

# Loading a quantized (4-bit) model with a per-process single-device device_map
# (device_map={"": local_rank}, used below for multi-GPU DDP) makes transformers
# expand that into one hf_device_map entry *per submodule* — all pointing at the
# same device, but accelerate's verify_device_map() only counts dict entries, so
# it mistakes this for `device_map="auto"` naive model-parallelism and refuses to
# wrap the model in DDP. This is accelerate's own documented escape hatch for
# exactly this case. Must be set before Trainer builds its Accelerator.
os.environ.setdefault("ACCELERATE_BYPASS_DEVICE_MAP", "true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--name-mapping", type=Path,
                        default=Path("data/anonymized/name_mapping.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/lora_adapter"))
    # Empirical (Phase 4 + tokenization probe): mean=104 tokens, p99=170, max=661.
    # 256 clips only 0.16% of examples and halves the padding budget vs 512.
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--max-train-examples", type=int, default=0,
                        help="Cap on training examples (0 = use all). Use a small "
                             "value (e.g. 1000) for a smoke test on a small GPU.")
    parser.add_argument("--max-val-examples", type=int, default=0,
                        help="Cap on validation examples (0 = use all).")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Optimization
    # Defaults are tuned for a 24 GB GPU (RTX 4090 / A40). For an 8 GB laptop GPU,
    # override with --batch-size 1 --grad-accum 16 --max-seq-length 256.
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Eval / save / log
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="ryGPT")
    parser.add_argument("--wandb-disabled", action="store_true")
    args = parser.parse_args()

    # Imports are local so the script remains importable without GPU deps installed.
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

    # ---- Tokenizer ----
    # We deliberately do NOT add <self> / <person_N> / <gf> / [media] etc. as
    # atomic special tokens. The base Qwen2.5 BPE already tokenizes them as
    # short multi-piece sequences (`<self>` → 3 tokens, `<person_N>` → 5).
    # Adding new tokens requires resizing the embedding + lm_head, and unless
    # we explicitly train those layers (which roughly doubles VRAM cost), the
    # new rows stay randomly initialized and the model emits garbage in their
    # positions — observed empirically as Thai script appearing at every
    # speaker prefix in the first smoke run.
    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Precision: bf16 (Ampere ≥ 8.0) or fp16 (Turing, e.g. T4) ----
    # torch.cuda.is_bf16_supported() returns True on T4 via software emulation on
    # newer PyTorch — misleading, so we check compute capability directly.
    if torch.cuda.is_available():
        cap_major, _ = torch.cuda.get_device_capability(0)
        use_bf16 = cap_major >= 8
        gpu_name = torch.cuda.get_device_name(0)
    else:
        use_bf16 = False
        gpu_name = "CPU"
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"GPU: {gpu_name}; precision: {'bf16' if use_bf16 else 'fp16'}")

    # ---- Model ----
    # device_map="auto" would split the model's layers *across* GPUs (naive
    # model parallelism) — only one GPU computes at a time, the rest sit idle.
    # For multi-GPU we instead want data parallelism: a full model copy per
    # process, each chewing through a different batch concurrently. When
    # launched via `accelerate launch --multi_gpu`, PartialState().process_index
    # gives each process its own GPU index; under a single process it's just 0.
    device_map = {"": PartialState().process_index}
    print(f"Loading base model in 4-bit: {args.base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model)

    # ---- LoRA ----
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
    model.print_trainable_parameters()

    # ---- Data ----
    print(f"Loading training data from {args.train_file}")
    train_examples = read_jsonl(args.train_file)
    val_examples = read_jsonl(args.val_file)
    print(f"  train={len(train_examples):,}  val={len(val_examples):,}")
    if args.max_train_examples > 0 and args.max_train_examples < len(train_examples):
        import random
        rng = random.Random(args.seed)
        train_examples = rng.sample(train_examples, args.max_train_examples)
        print(f"  --max-train-examples: subsampled to {len(train_examples):,}")
    if args.max_val_examples > 0 and args.max_val_examples < len(val_examples):
        import random
        rng = random.Random(args.seed)
        val_examples = rng.sample(val_examples, args.max_val_examples)
        print(f"  --max-val-examples: subsampled to {len(val_examples):,}")

    def tokenize_one(example: dict) -> dict:
        messages = example_to_chat_messages(example)
        # Full sequence (everything we want to encode).
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        full_ids = tokenizer(
            full_text, truncation=True, max_length=args.max_seq_length,
            add_special_tokens=False,
        )["input_ids"]

        # Prefix is everything except the final assistant turn, plus the
        # generation prompt that signals "model speaks next".
        prefix_messages = messages[:-1]
        prefix_text = tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True,
        )
        prefix_ids = tokenizer(
            prefix_text, truncation=True, max_length=args.max_seq_length,
            add_special_tokens=False,
        )["input_ids"]

        labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
        # Defensive: if truncation broke alignment, just drop this example
        # (we'll filter empty-labels rows below).
        if len(labels) != len(full_ids):
            labels = [-100] * len(full_ids)
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }

    train_ds = Dataset.from_list(train_examples).map(
        tokenize_one, remove_columns=Dataset.from_list(train_examples).column_names,
        desc="Tokenizing train",
    )
    val_ds = Dataset.from_list(val_examples).map(
        tokenize_one, remove_columns=Dataset.from_list(val_examples).column_names,
        desc="Tokenizing val",
    )
    # Drop rows where all labels are -100 (truncation killed the target).
    train_ds = train_ds.filter(lambda r: any(l != -100 for l in r["labels"]))
    val_ds = val_ds.filter(lambda r: any(l != -100 for l in r["labels"]))
    print(f"  after filter: train={len(train_ds):,}  val={len(val_ds):,}")

    # ---- Trainer ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_to = "wandb" if not args.wandb_disabled else "none"
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
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        report_to=report_to,
        run_name=f"qlora-{args.base_model.split('/')[-1]}",
    )

    # DataCollatorForSeq2Seq correctly pads input_ids/attention_mask AND labels
    # (padding labels with -100). DataCollatorForLanguageModeling would either
    # overwrite our masked labels (SFT-style) or fail to pad them at all.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Persist training config for the eval phase to read back.
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

    # Resume automatically if out_dir already has a checkpoint (e.g. a prior
    # Kaggle session that ran out of time). Training args, LoRA config, and
    # seed must match the original run for this to resume cleanly.
    last_checkpoint = get_last_checkpoint(str(args.out_dir)) if args.out_dir.exists() else None
    if last_checkpoint:
        print(f"\nFound existing checkpoint — resuming from: {last_checkpoint}")
    else:
        print("\nStarting training …")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(args.out_dir))
    tokenizer.save_pretrained(args.out_dir)
    print(f"\nSaved adapter + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
