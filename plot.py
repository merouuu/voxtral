"""Main figure: probe accuracy along Voxtral (solid, ± std over actor splits) vs zero-shot behavior (dashed).

Usage: .venv/bin/python plot.py outputs
"""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOOKS = ["encoder", "adapter", "L0", "L4", "L8", "L12", "L16", "L20", "L24", "L29"]
XLABELS = ["Whisper\n50 Hz", "Adapter\n12.5 Hz", "L0", "L4", "L8", "L12", "L16", "L20", "L24", "L29"]
TASKS = {"text": ("Text (12 sentences)", "#2f6db0"), "gender": ("Gender (2)", "#b07d12"), "emotion": ("Emotion (6)", "#c23a5c")}
HUMAN_EMOTION = (0.41 - 1 / 6) / (1 - 1 / 6)  # audio-only human raters, Cao et al. 2014

out = Path(sys.argv[1])
probes = {(r["hook"], r["task"]): r for r in csv.DictReader(open(out / "probe_results.csv"))}
behavior = {(r["task"], r["variant"]): float(r["acc_norm"]) for r in csv.DictReader(open(out / "behavioral_results.csv"))}

fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(HOOKS))
for task, (label, color) in TASKS.items():
    m = [100 * float(probes[h, task]["acc_norm"]) for h in HOOKS]
    s = [100 * float(probes[h, task]["std"]) for h in HOOKS]
    ax.plot(x, m, color=color, lw=2.2, marker="o", ms=4, label=f"{label}, linear probe")
    ax.fill_between(x, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)], color=color, alpha=0.15, lw=0)
    b = 100 * behavior[task, "letter"]
    ax.axhline(b, color=color, ls="--", lw=1.5)
    ax.text(0, b + 1.5, f"{label.split(' ')[0]}, model answer: {b:.0f} %", color=color, fontsize=9)

ax.axhline(100 * HUMAN_EMOTION, color="#98a1b2", ls=":", lw=1)
ax.text(0, 100 * HUMAN_EMOTION + 1.5, "human raters, emotion (audio only)", color="#5b6475", fontsize=8)

emo_end = 100 * float(probes["L29", "emotion"]["acc_norm"])
emo_beh = 100 * behavior["emotion", "letter"]
ax.annotate("", xy=(len(HOOKS) - 1.25, emo_beh + 1), xytext=(len(HOOKS) - 1.25, emo_end - 1),
            arrowprops=dict(arrowstyle="<->", color="#c23a5c", lw=1.5))
ax.text(len(HOOKS) - 1.4, (emo_end + emo_beh) / 2, "probe-behavior\ngap", color="#c23a5c", ha="right", va="center", fontsize=10)

ax.set_xticks(list(x), XLABELS)
ax.set_xlabel("Probe point along Voxtral-Mini-3B (decoder layer residual at audio positions)")
ax.set_ylabel("Balanced accuracy, chance-normalized (%)")
ax.set_ylim(-5, 105)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="center left", frameon=False, fontsize=9)
ax.set_title("Voxtral encodes emotion internally but answers at chance (CREMA-D, unseen actors)", fontsize=11)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(out / f"main_figure.{ext}", dpi=300)
print("saved", out / "main_figure.png")
