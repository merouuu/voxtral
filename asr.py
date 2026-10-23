"""ASR check: does the LoRA hurt free transcription? Test actors (seed 0), Voxtral's standard transcription
request, greedy decoding, without then with the LoRA. WER and exact-sentence rate vs the CREMA-D sentences.

Usage: GPU=1 ./run.sh asr.py outputs [n_clips]   -> outputs/asr.csv, outputs/asr_results.csv
"""
import csv, re, sys
from pathlib import Path
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration
from data import SENTENCES, split_by_actor
from lora import add_lora
from progress import track

MODEL = "mistralai/Voxtral-Mini-3B-2507"
norm = lambda s: re.sub(r"[^a-z' ]", " ", s.lower()).split()

def word_errors(ref, hyp):  # Levenshtein distance on words
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1]

out = Path(sys.argv[1])
meta = list(csv.DictReader(open(out / "meta.csv")))
_, test = split_by_actor([dict(m, actor=int(m["actor"])) for m in meta], seed=0)
test = test[: int(sys.argv[2]) if len(sys.argv) > 2 else None]

processor = AutoProcessor.from_pretrained(MODEL)
model = VoxtralForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

rows, summary = [], [("model", "wer", "exact_sentence")]
for name in ("base", "lora"):
    if name == "lora":
        add_lora(model, out / "lora.pt")
    errors = words = exact = 0
    for c in track(test, f"asr {name}"):
        inp = processor.apply_transcription_request(audio=c["path"], model_id=MODEL, language="en").to("cuda", torch.bfloat16)
        with torch.inference_mode():
            ids = model.generate(**inp, max_new_tokens=40, do_sample=False)
        hyp = processor.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        ref = SENTENCES[c["sentence"]]
        e = word_errors(norm(ref), norm(hyp))
        errors, words, exact = errors + e, words + len(norm(ref)), exact + (e == 0)
        rows.append((name, c["path"], ref, hyp, e))
    summary.append((name, round(errors / words, 4), round(exact / len(test), 4)))
    print(summary[-1], flush=True)

with open(out / "asr.csv", "w", newline="") as f:
    csv.writer(f).writerows([("model", "path", "ref", "hyp", "word_errors"), *rows])
with open(out / "asr_results.csv", "w", newline="") as f:
    csv.writer(f).writerows(summary)
print(*summary, sep="\n")
