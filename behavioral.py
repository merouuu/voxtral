"""Bloc 0: zero-shot forced choice on the test actors, to measure the probe-behavior gap.

Two variants, same shuffled option order per clip:
  letter: options listed as A) B) ..., prediction = argmax of the next-token logits over the valid letters
  word:   options listed by name, prediction = option with the highest length-normalized log-likelihood
  open:   emotion only, a phrasing never seen in LoRA training and no options listed, scored like word

Usage: GPU=1 ./run.sh behavioral.py outputs [n_clips]
       LORA=outputs/lora.pt GPU=1 ./run.sh behavioral.py outputs   (writes *_lora.csv)
       VARIANTS=open GPU=1 ./run.sh behavioral.py outputs            (only these variants, writes *_open.csv)
"""
import csv, os, random, sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from transformers import AutoProcessor, VoxtralForConditionalGeneration
from data import EMOTIONS, SENTENCES, split_by_actor
from progress import track

MODEL = "mistralai/Voxtral-Mini-3B-2507"
TASKS = {  # task: (question, meta column, labels)
    "emotion": ("What emotion does the speaker express?", "emotion", list(EMOTIONS.values())),
    "gender": ("Is the speaker male or female?", "gender", ["male", "female"]),
    "text": ("Which sentence was spoken?", "sentence", list(SENTENCES)),
}
OPEN_EMOTION = "How would you describe the speaker's emotional state?"
LETTERS = "ABCDEFGHIJKL"
text_of = lambda label: SENTENCES.get(label, label)

out = Path(sys.argv[1])
meta = list(csv.DictReader(open(out / "meta.csv")))
_, test = split_by_actor([dict(m, actor=int(m["actor"])) for m in meta], seed=0)  # same test actors as probe seed 0
test = test[: int(sys.argv[2]) if len(sys.argv) > 2 else None]

processor = AutoProcessor.from_pretrained(MODEL)
tok = processor.tokenizer
model = VoxtralForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
suffix = ""
if os.environ.get("LORA"):
    from lora import add_lora
    add_lora(model, os.environ["LORA"])
    suffix = "_lora"
VARIANTS = os.environ.get("VARIANTS", "letter,word,open").split(",")
if "VARIANTS" in os.environ:
    suffix += "_" + "-".join(VARIANTS)
letter_ids = [tok.encode(L, add_special_tokens=False) for L in LETTERS]
assert all(len(t) == 1 for t in letter_ids), letter_ids
letter_ids = [t[0] for t in letter_ids]

def user_turn(path, prompt):
    conv = [{"role": "user", "content": [{"type": "audio", "path": path}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(conv, return_tensors="pt").to("cuda", torch.bfloat16)

@torch.inference_mode()
def letter_choice(path, question, options):
    prompt = f"{question} " + " ".join(f"{L}) {text_of(o)}" for L, o in zip(LETTERS, options)) + " Answer with the letter only."
    h = model.model(**user_turn(path, prompt), use_cache=False).last_hidden_state[0, -1]
    return options[model.lm_head(h)[letter_ids[: len(options)]].argmax().item()]

@torch.inference_mode()
def word_choice(path, question, options, list_options=True):
    if list_options:
        prompt = f"{question} Options: " + "; ".join(text_of(o) for o in options) + ". Answer with one of the options only."
    else:
        prompt = f"{question} Answer with one word."
    inp = user_turn(path, prompt)
    prefix = inp["input_ids"][0]
    answers = [torch.tensor(tok.encode(text_of(o), add_special_tokens=False), device="cuda") for o in options]
    n, L = len(options), len(prefix) + max(len(a) for a in answers)
    ids = torch.zeros(n, L, dtype=torch.long, device="cuda")
    mask = torch.zeros(n, L, dtype=torch.long, device="cuda")
    for j, a in enumerate(answers):  # right padding: the shared prefix keeps the same positions in every row
        ids[j, : len(prefix) + len(a)] = torch.cat([prefix, a])
        mask[j, : len(prefix) + len(a)] = 1
    feats = inp["input_features"].repeat(n, *[1] * (inp["input_features"].dim() - 1))
    h = model.model(input_ids=ids, attention_mask=mask, input_features=feats, use_cache=False).last_hidden_state
    scores = []
    for j, a in enumerate(answers):
        pos = torch.arange(len(prefix) - 1, len(prefix) - 1 + len(a), device="cuda")  # logits predicting each answer token
        logp = torch.log_softmax(model.lm_head(h[j, pos]).float(), -1)
        scores.append(logp.gather(1, a[:, None]).mean().item())
    return options[int(np.argmax(scores))]

rows = []
for i, c in enumerate(track(test, f"behavioral{suffix}")):
    for task, (question, col, labels) in TASKS.items():
        options = random.Random(i).sample(labels, len(labels))  # shuffled order, shared by both variants
        if "letter" in VARIANTS:
            rows.append((c["path"], task, "letter", c[col], letter_choice(c["path"], question, options)))
        if "word" in VARIANTS:
            rows.append((c["path"], task, "word", c[col], word_choice(c["path"], question, options)))
        if task == "emotion" and "open" in VARIANTS:
            rows.append((c["path"], task, "open", c[col], word_choice(c["path"], OPEN_EMOTION, options, list_options=False)))

with open(out / f"behavioral{suffix}.csv", "w", newline="") as f:
    csv.writer(f).writerows([("path", "task", "variant", "true", "pred"), *rows])

probes = {(r["hook"], r["task"]): float(r["acc_norm"]) for r in csv.DictReader(open(out / "probe_results.csv"))}
summary = [("task", "variant", "acc", "acc_norm", "probe_encoder_norm", "probe_L29_norm")]
for task, (_, _, labels) in TASKS.items():
    for variant in ("letter", "word", "open"):
        y = [(t, p) for _, tk, v, t, p in rows if tk == task and v == variant]
        if not y:
            continue
        acc = balanced_accuracy_score(*zip(*y))
        chance = 1 / len(labels)
        summary.append((task, variant, round(acc, 4), round((acc - chance) / (1 - chance), 4),
                        probes.get(("encoder", task)), probes.get(("L29", task))))
        if task == "emotion":
            cm = confusion_matrix(*zip(*y), labels=labels, normalize="true").round(3)
            with open(out / f"confusion_emotion_{variant}{suffix}.csv", "w", newline="") as f:
                csv.writer(f).writerows([["true\\pred", *labels], *[[l, *r] for l, r in zip(labels, cm)]])
            print(f"emotion confusion ({variant}), rows = true:\n", labels, "\n", cm)
with open(out / f"behavioral_results{suffix}.csv", "w", newline="") as f:
    csv.writer(f).writerows(summary)
print(*summary, sep="\n")
