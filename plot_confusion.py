"""Emotion confusion on the unseen question, without vs with LoRA (article style).

Usage: python plot_confusion.py outputs [en|fr] [--no-tex]   -> figures/confusion_emotion_en.*
"""
import csv, sys
from pathlib import Path
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import article_style as S

out = Path(sys.argv[1])
lang = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "en"
S.setup_style(lang=lang, usetex="--no-tex" not in sys.argv)
WORDS = {"fr": {"base": "sans LoRA", "lora": "avec LoRA", "x": "émotion répondue", "y": "émotion jouée", "bar": "part de la ligne"},
         "en": {"base": "without LoRA", "lora": "with LoRA", "x": "predicted emotion", "y": "true emotion", "bar": "share of row"}}[lang]
PANELS = {"base": "confusion_emotion_open_open.csv", "lora": "confusion_emotion_open_lora.csv"}

def load(name):
    rows = list(csv.reader(open(out / name)))
    return rows[0][1:], np.array([[float(v) for v in r[1:]] for r in rows[1:]])

cmap = LinearSegmentedColormap.from_list("emo", ["#ffffff", S.ORANGE])
fig = S.figure(2.2, width=5.2)
axes = fig.subplots(1, 2, sharey=True)
for ax, (key, name) in zip(axes, PANELS.items()):
    labels, m = load(name)
    im = ax.imshow(100 * m, cmap=cmap, vmin=0, vmax=100)
    for i in range(len(labels)):
        for j in range(len(labels)):
            S.label(ax, j, i, f"{100 * m[i, j]:.0f}", ha="center", va="center", fontsize=S.FS - 1.5,
                    color="white" if m[i, j] > 0.55 else S.INK)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(S.tex(WORDS[key]), fontsize=S.FS)
    ax.set_xlabel(S.tex(WORDS["x"]))
axes[0].set_ylabel(S.tex(WORDS["y"]))
bar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03, format=S.pct(0))
bar.set_label(S.tex(WORDS["bar"]))
S.save(fig, "confusion_emotion", out_dir="figures")
