"""Phase 8 (optional) — local Gradio demo.

GATED on Phase 7 memorization audit. If `eval/memorization.md` exists and shows
the gate as FAILED, this script refuses to start. Override only with --skip-gate.

Local-only by default (127.0.0.1). Never sets share=True. Do not expose publicly.

Reads:
  models/lora_adapter/
  eval/memorization.md  (gate)

Run:
    python scripts/08_demo.py
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path


def _check_memorization_gate(memorization_md: Path) -> tuple[bool, str]:
    if not memorization_md.exists():
        return False, (
            f"{memorization_md} not found. Run Phase 7 first; the demo is gated "
            "on the memorization audit."
        )
    text = memorization_md.read_text(encoding="utf-8")
    if "FAILED — do not deploy demo" in text:
        return False, (
            "Memorization audit FAILED (see eval/memorization.md). Demo blocked. "
            "Retrain with stronger regularization or use --skip-gate to override."
        )
    return True, "Memorization gate passed."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-model", type=str, default=None,
                        help="Override; defaults to value from training_config.json")
    parser.add_argument("--adapter-dir", type=Path, default=Path("models/lora_adapter"))
    parser.add_argument("--memorization-md", type=Path,
                        default=Path("eval/memorization.md"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--skip-gate", action="store_true",
                        help="Bypass the memorization-audit gate (NOT RECOMMENDED).")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="1.0 = none, 1.2 = mild (suppresses 'Aaah Aah Aaah' loops)")
    args = parser.parse_args()

    # ---- Gate ----
    if not args.skip_gate:
        passed, msg = _check_memorization_gate(args.memorization_md)
        if not passed:
            raise SystemExit(msg)
        print(msg)

    try:
        import gradio as gr
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            f"Missing dep ({e.name}). Install: pip install gradio + the GPU training stack."
        )

    # ---- Resolve base model from training config ----
    cfg_path = args.adapter_dir / "training_config.json"
    if args.base_model is None:
        if not cfg_path.exists():
            raise SystemExit(
                f"--base-model not given and {cfg_path} not found."
            )
        with cfg_path.open(encoding="utf-8") as f:
            args.base_model = json.load(f)["base_model"]

    from src.dataset import chat_stop_token_ids
    from src.eval import structural_stopping_criteria

    # ---- Load ----
    print(f"Loading tokenizer from {args.adapter_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = chat_stop_token_ids(tokenizer)

    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    else:
        use_bf16 = False
    load_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Loading base model: {args.base_model} ({'bf16' if use_bf16 else 'fp16'}) …")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=load_dtype, device_map="auto",
    )

    print(f"Attaching LoRA adapter from {args.adapter_dir} …")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    SELF = "<self>"

    def respond(relationship: str, other_speaker: str, incoming_message: str) -> str:
        """Single-turn generation: given a fresh incoming message, produce a reply."""
        if not incoming_message.strip():
            return ""
        rel_token = f"<{relationship}>"
        speaker = other_speaker.strip() or "<person_1>"
        messages = [
            {"role": "system", "content": rel_token},
            {"role": "user", "content": f"{speaker}: {incoming_message.strip()}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            next(model.parameters()).device
        )
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=stop_ids,
                stopping_criteria=structural_stopping_criteria(tokenizer, prompt_len),
            )
        new_tokens = out[0, prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Strip any leading speaker prefix the model emits.
        if text.lower().startswith(f"{SELF.lower()}:"):
            text = text.split(":", 1)[1].strip()
        return text

    iface = gr.Interface(
        fn=respond,
        inputs=[
            gr.Radio(["gf", "friend", "group"], value="gf", label="Relationship"),
            gr.Textbox(value="<person_3>", label="Other speaker token "
                       "(e.g. <person_1> .. <person_5>; matters most for group)"),
            gr.Textbox(label="Incoming message (in Manglish)", lines=2),
        ],
        outputs=gr.Textbox(label="Reply (in my style)", lines=2),
        title="ryGPT — Manglish personal LM",
        description=(
            "Local-only demo. Type as if you were the other person in the chat; "
            "the model replies as me. Gated on the Phase 7 memorization audit."
        ),
        flagging_mode="never",
    )

    print(f"Serving on http://{args.host}:{args.port}")
    iface.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
