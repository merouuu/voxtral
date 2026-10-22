"""Step 2: a small LoRA on the top decoder layers, trained to *say* the emotion it already encodes.

Train actors of split seed 0 only. Each example: audio, then a question, then a one-word answer;
next-token loss on the answer only. Varied phrasings (none of them used by behavioral.py),
mixed with transcription examples so ASR is not forgotten.

Usage: GPU=1 ./run.sh lora.py outputs [n_examples]   -> outputs/lora.pt
"""
import csv, random, sys
from pathlib import Path
import torch

LAYERS, RANK, ALPHA = range(20, 30), 8, 16
TARGETS = ("q_proj", "v_proj")


class LoRA(torch.nn.Module):
    """base(x) + scale * B A x, with A random and B zero so training starts from the original model."""
    def __init__(self, base, r=RANK, alpha=ALPHA):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / base.in_features ** 0.5)
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (x.float() @ self.A.T @ self.B.T * self.scale).to(x.dtype)


def add_lora(model, path=None):
    """Freeze the model, wrap the target projections, optionally load trained weights; returns the LoRA params."""
    model.requires_grad_(False)
    for l in LAYERS:
        attn = model.model.language_model.layers[l].self_attn
        for name in TARGETS:
            setattr(attn, name, LoRA(getattr(attn, name)))
    if path:
        missing = model.load_state_dict(torch.load(path), strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
    return [p for n, p in model.named_parameters() if n.endswith((".A", ".B"))]


if __name__ == "__main__":
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    from data import SENTENCES, split_by_actor

    MODEL = "mistralai/Voxtral-Mini-3B-2507"
    EMOTION_QUESTIONS = [
        "Which emotion is conveyed in this recording?", "What is the emotional tone of the voice?",
        "Identify the speaker's emotion.", "How does the speaker sound emotionally?",
        "Classify the emotion in this speech.",
    ]
    ASR_QUESTIONS = ["Repeat exactly what is said.", "Transcribe this audio."]

    out = Path(sys.argv[1])
    meta = list(csv.DictReader(open(out / "meta.csv")))
    train, _ = split_by_actor([dict(m, actor=int(m["actor"])) for m in meta], seed=0)
    rng = random.Random(0)
    examples = [(c["path"], rng.choice(EMOTION_QUESTIONS), c["emotion"]) for c in train]
    examples += [(c["path"], rng.choice(ASR_QUESTIONS), SENTENCES[c["sentence"]]) for c in rng.sample(train, len(train) // 3)]
    rng.shuffle(examples)
    examples = examples[: int(sys.argv[2]) if len(sys.argv) > 2 else None]

    processor = AutoProcessor.from_pretrained(MODEL)
    tok = processor.tokenizer
    model = VoxtralForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    params = add_lora(model)
    print(f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable params, {len(examples)} examples", flush=True)
    opt = torch.optim.AdamW(params, lr=2e-4)
    ACCUM = 8

    running = 0.0
    for step, (path, question, answer) in enumerate(examples, 1):
        conv = [{"role": "user", "content": [{"type": "audio", "path": path}, {"type": "text", "text": question}]}]
        inp = processor.apply_chat_template(conv, return_tensors="pt").to("cuda", torch.bfloat16)
        target = torch.tensor(tok.encode(answer, add_special_tokens=False) + [tok.eos_token_id], device="cuda")
        n = inp["input_ids"].shape[1]
        ids = torch.cat([inp["input_ids"][0], target])[None]
        h = model.model(input_ids=ids, input_features=inp["input_features"], use_cache=False).last_hidden_state
        logits = model.lm_head(h[0, n - 1 : -1]).float()  # positions predicting the answer tokens
        loss = torch.nn.functional.cross_entropy(logits, target)
        (loss / ACCUM).backward()
        running += loss.item()
        if step % ACCUM == 0:
            opt.step()
            opt.zero_grad()
        if step % 400 == 0:
            print(f"{step}/{len(examples)} loss {running / 400:.3f}", flush=True)
            running = 0.0

    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith((".A", ".B"))}, out / "lora.pt")
    print("saved", out / "lora.pt")
