"""One Voxtral forward per CREMA-D clip -> mean-pooled activations per hook, with and without the padding mask (Q8).

Usage: GPU=1 ./run.sh extract.py data/CREMA-D outputs [n_clips]
"""
import csv, math, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration
from data import load_clips

MODEL = "mistralai/Voxtral-Mini-3B-2507"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 29]
PROMPT = "Describe this audio."

root, out = sys.argv[1], Path(sys.argv[2])
clips = load_clips(root)[: int(sys.argv[3]) if len(sys.argv) > 3 else None]
out.mkdir(parents=True, exist_ok=True)

processor = AutoProcessor.from_pretrained(MODEL)
model = VoxtralForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
audio_id = model.config.audio_token_id

acts = {}
def save(name):
    def hook(module, inputs, output):
        o = output[0] if isinstance(output, tuple) else getattr(output, "last_hidden_state", output)
        acts[name] = o.reshape(-1, o.shape[-1])  # drop batch dim
    return hook
model.model.audio_tower.register_forward_hook(save("encoder"))
model.model.multi_modal_projector.register_forward_hook(save("adapter"))
for l in LAYERS:
    model.model.language_model.layers[l].register_forward_hook(save(f"L{l}"))

feats = {}
for i, c in enumerate(clips):
    conv = [{"role": "user", "content": [{"type": "audio", "path": str(c["path"])}, {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(conv, return_tensors="pt").to("cuda", torch.bfloat16)
    with torch.inference_mode():
        model.model(**inputs, use_cache=False)
    c["duration"] = sf.info(c["path"]).duration
    audio_pos = (inputs["input_ids"][0] == audio_id).nonzero().squeeze(1)
    for name, a in acts.items():
        if name.startswith("L"):
            a = a[audio_pos]
        # encoder runs at 50 Hz, adapter and decoder at 12.5 Hz; keep only frames covering real audio
        n = min(len(a), math.ceil(c["duration"] * (50 if name == "encoder" else 12.5)))
        feats.setdefault(name, []).append(a[:n].float().mean(0).cpu().numpy())
        feats.setdefault(f"{name}_all", []).append(a.float().mean(0).cpu().numpy())
    if i % 500 == 0:
        print(f"{i}/{len(clips)}", flush=True)

np.savez(out / "features.npz", **{k: np.stack(v).astype(np.float16) for k, v in feats.items()})
with open(out / "meta.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path", "actor", "sentence", "emotion", "gender", "duration"])
    w.writeheader()
    w.writerows(clips)
print("saved", {k: len(v[0]) for k, v in feats.items()}, "for", len(clips), "clips")
