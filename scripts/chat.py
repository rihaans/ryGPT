"""Interactive multi-turn chat with a trained adapter.

Maintains a running conversation history so each reply sees the previous
exchanges as context — mirrors how the model was trained.

Run:
    python scripts/chat.py --adapter-dir models/lora_adapter_smoke
    python scripts/chat.py --adapter-dir models/lora_adapter_smoke --relationship friend

Commands during the chat:
    /reset      clear conversation history (keep model loaded)
    /history    show current conversation
    /rel <x>    switch relationship (gf / friend / group)
    /speaker <x> switch the other speaker label
    /quit       exit
"""
import _bootstrap  # noqa: F401

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter-dir", type=Path, default=Path("models/lora_adapter_smoke"))
    p.add_argument("--base-model", type=str, default=None,
                   help="Override; defaults to value from training_config.json")
    p.add_argument("--relationship", choices=["gf", "friend", "group"], default="gf")
    p.add_argument("--other-speaker", default="<person_3>",
                   help="Token for the other person you're chatting as (e.g. <person_3>)")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.2,
                   help="Penalty for repeating tokens (1.0 = no penalty, 1.2 = mild).")
    p.add_argument("--max-history-turns", type=int, default=8,
                   help="Keep at most N recent turns in context.")
    args = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(f"Missing dep ({e.name}). Install the training stack first.")

    # Resolve base model from training_config.json if not provided
    if args.base_model is None:
        cfg_path = args.adapter_dir / "training_config.json"
        if not cfg_path.exists():
            raise SystemExit(
                f"--base-model not given and {cfg_path} not found."
            )
        args.base_model = json.loads(cfg_path.read_text(encoding="utf-8"))["base_model"]

    print(f"Loading tokenizer + base model ({args.base_model}) …")
    tok = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    load_dtype = torch.bfloat16 if use_bf16 else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=load_dtype, device_map="auto",
    )
    print(f"Attaching adapter from {args.adapter_dir} …")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    device = next(model.parameters()).device

    print()
    print("=" * 60)
    print(f"  ryGPT chat — relationship: {args.relationship}, "
          f"other: {args.other_speaker}")
    print(f"  Type /quit to exit, /reset to clear history, /history to view,")
    print(f"  /rel <gf|friend|group>, /speaker <token>")
    print("=" * 60)

    history: list[tuple[str, str]] = []  # list of (speaker_token, text)

    def render_messages() -> list[dict]:
        """Build the chat template messages list from current state."""
        msgs = [{"role": "system", "content": f"<{args.relationship}>"}]
        # Group consecutive same-speaker entries into one turn (like training).
        grouped: list[tuple[str, list[str]]] = []
        for sp, txt in history[-args.max_history_turns:]:
            if grouped and grouped[-1][0] == sp:
                grouped[-1][1].append(txt)
            else:
                grouped.append((sp, [txt]))
        for sp, texts in grouped:
            role = "assistant" if sp == "<self>" else "user"
            msgs.append({"role": role, "content": f"{sp}: " + "\n".join(texts)})
        return msgs

    def respond(user_text: str) -> str:
        history.append((args.other_speaker, user_text))
        messages = render_messages()
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True).strip()
        # Strip leading "<self>:" speaker prefix the model emits
        for prefix in ("<self>:", "<self> :", "<self>"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].lstrip(":").strip()
                break
        history.append(("<self>", text))
        return text

    while True:
        try:
            user_in = input(f"\n[{args.other_speaker}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_in:
            continue
        if user_in == "/quit":
            break
        if user_in == "/reset":
            history.clear()
            print("(history cleared)")
            continue
        if user_in == "/history":
            for sp, txt in history:
                print(f"  {sp}: {txt}")
            continue
        if user_in.startswith("/rel "):
            new = user_in.split()[1].strip().lower()
            if new in ("gf", "friend", "group"):
                args.relationship = new
                print(f"(relationship -> {new})")
            else:
                print("(must be gf / friend / group)")
            continue
        if user_in.startswith("/speaker "):
            args.other_speaker = user_in.split(maxsplit=1)[1].strip()
            print(f"(other speaker -> {args.other_speaker})")
            continue

        reply = respond(user_in)
        print(f"[<self>] {reply}")


if __name__ == "__main__":
    main()
