"""Main figure (article style): probe accuracy along Voxtral (solid, ± std over actor splits)
vs what the model answers zero-shot (dashed). The vertical distance on emotion is the probe-behavior gap.

Usage: python plot.py outputs [en|fr] [--no-tex]   (English by default -> figures/main_figure_en.*)
"""
import csv, sys
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import article_style as S

out = Path(sys.argv[1])
lang = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "en"
S.setup_style(lang=lang, usetex="--no-tex" not in sys.argv)

HOOKS = ["encoder", "adapter", "L0", "L4", "L8", "L12", "L16", "L20", "L24", "L29"]
TASKS = {"text": S.BLUE, "gender": S.AQUA, "emotion": S.ORANGE}
WORDS = {
    "fr": {"text": "phrase prononcée (12 classes)", "gender": "genre (2 classes)", "emotion": "émotion (6 classes)",
           "probe": "sonde linéaire (± é.-t., 5 splits)", "answer": "réponse du modèle (zéro-shot)", "lora": "réponse après LoRA (question inédite)", "x": "point de sonde (encodeur, adaptateur, couche du décodeur)",
           "y": "acc. équilibrée (norm.)", "enc": "enc.", "adapt": "ad."},
    "en": {"text": "spoken sentence (12 classes)", "gender": "gender (2 classes)", "emotion": "emotion (6 classes)",
           "probe": "linear probe (± std, 5 splits)", "answer": "model answer (zero-shot)", "lora": "answer after LoRA (unseen question)", "x": "probe point (encoder, adapter, decoder layer)",
           "y": "balanced accuracy (norm.)", "enc": "enc.", "adapt": "ad."},
}[lang]

probes = {(r["hook"], r["task"]): r for r in csv.DictReader(open(out / "probe_results.csv"))}
behavior = {(r["task"], r["variant"]): 100 * float(r["acc_norm"]) for r in csv.DictReader(open(out / "behavioral_results.csv"))}
lora = {(r["task"], r["variant"]): 100 * float(r["acc_norm"]) for r in csv.DictReader(open(out / "behavioral_results_lora.csv"))}

fig = S.figure(2.3)
ax = fig.add_subplot()
x = list(range(len(HOOKS)))
for task, color in TASKS.items():
    m = [100 * float(probes[h, task]["acc_norm"]) for h in HOOKS]
    s = [100 * float(probes[h, task]["std"]) for h in HOOKS]
    ax.plot(x, m, color=color, marker="o")
    ax.fill_between(x, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)], color=color, alpha=0.18, lw=0)
for task, color in TASKS.items():
    ax.axhline(behavior[task, "letter"], color=color, ls=S.DASH, lw=1.1)
ax.axhline(lora["emotion", "open"], color=S.ORANGE, ls="-.", lw=1.1)  # LoRA on decoder layers 20-29, phrasing unseen in training

pct_text = lambda v: f"{v:.0f} %" if lang == "fr" else f"{v:.0f}%"  # S.label escapes it for LaTeX

# probe-behavior gap on emotion: last-layer probe value vs model answer (values only, the legend names the series)
top, bottom = 100 * float(probes["L29", "emotion"]["acc_norm"]), behavior["emotion", "letter"]
S.label(ax, x[-1] + 0.15, top + 3, pct_text(top), ha="center", va="bottom", color=S.ORANGE)
S.label(ax, x[0], bottom + 2.5, pct_text(bottom), ha="left", va="bottom", color=S.ORANGE)
S.label(ax, x[0], lora["emotion", "open"] + 2.5, pct_text(lora["emotion", "open"]), ha="left", va="bottom", color=S.ORANGE)

ax.set_xticks(x, [WORDS["enc"], WORDS["adapt"], *[h[1:] for h in HOOKS[2:]]])
ax.set_xlim(-0.4, x[-1] + 0.4)
ax.set_ylim(-5, 105)
ax.yaxis.set_major_formatter(S.pct(0))
ax.set_xlabel(S.tex(WORDS["x"]))
ax.set_ylabel(S.tex(WORDS["y"]))
# two boxes: color = task, line style = what is measured
tasks = [Patch(color=c, label=S.tex(WORDS[t])) for t, c in TASKS.items()]
styles = [Line2D([], [], color=S.INK, marker="o", label=S.tex(WORDS["probe"])),
          Line2D([], [], color=S.INK, ls=S.DASH, label=S.tex(WORDS["answer"])),
          Line2D([], [], color=S.INK, ls="-.", label=S.tex(WORDS["lora"]))]
ax.add_artist(S.legend_out(ax, handles=tasks))
S.legend_out(ax, handles=styles, loc="lower left", bbox_to_anchor=(1.03, 0.0))
S.save(fig, "main_figure", out_dir="figures")
