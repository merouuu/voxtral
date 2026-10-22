"""J0 smoke test: one Voxtral forward on one CREMA-D clip, print module names and hook shapes."""
import sys, torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

MODEL = "mistralai/Voxtral-Mini-3B-2507"
clip = sys.argv[1] if len(sys.argv) > 1 else "data/CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav"

processor = AutoProcessor.from_pretrained(MODEL)
model = VoxtralForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
print(model.model.multi_modal_projector, len(model.model.language_model.layers), "decoder layers")
print("audio_token_id:", model.config.audio_token_id)

shapes = {}
def hook(name):
    return lambda m, i, o: shapes.__setitem__(name, tuple((o[0] if isinstance(o, tuple) else getattr(o, "last_hidden_state", o)).shape))
model.model.audio_tower.register_forward_hook(hook("encoder"))
model.model.multi_modal_projector.register_forward_hook(hook("adapter"))

conv = [{"role": "user", "content": [{"type": "audio", "path": clip}, {"type": "text", "text": "Describe this audio."}]}]
inputs = processor.apply_chat_template(conv, return_tensors="pt").to("cuda", torch.bfloat16)
print("n audio tokens:", (inputs["input_ids"] == model.config.audio_token_id).sum().item())
out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
print("shapes:", shapes)
print("answer:", processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
