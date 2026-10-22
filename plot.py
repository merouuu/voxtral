"""Main figure (article style): probe accuracy along Voxtral (solid, ± std over actor splits)
vs what the model answers zero-shot (dashed). The vertical distance on emotion is the probe-behavior gap.

Usage: python plot.py outputs [fr|en] [--no-tex]
"""
import csv, sys
from pathlib import Path
import article_style as S

out = Path(sys.argv[1])
lang = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "fr"
S.setup_style(lang=lang, usetex="--no-tex" not in sys.argv)

HOOKS = ["encoder", "adapter", "L0", "L4", "L8", "L12", "L16", "L20", "L24", "L29"]
TASKS = {"text": S.BLUE, "gender": S.AQUA, "emotion": S.ORANGE}
WORDS = {
    "fr": {"text": "texte (12 phrases)", "gender": "genre (2)", "emotion": "émotion (6)", "probe": "sonde", "answer": "réponse",
           "human": "humains, émotion", "x": "point de sonde (encodeur, adaptateur, couche du décodeur)",
           "y": "accuracy équilibrée, normalisée", "enc": "enc.", "adapt": "ad."},
    "en": {"text": "text (12 sentences)", "gender": "gender (2)", "emotion": "emotion (6)", "probe": "probe", "answer": "answer",
           "human": "humans, emotion", "x": "probe point (encoder, adapter, decoder layer)",
           "y": "balanced accuracy, chance-normalized", "enc": "enc.", "adapt": "ad."},
}[lang]
HUMAN_EMOTION = 100 * (0.41 - 1 / 6) / (1 - 1 / 6)  # audio-only human raters, Cao et al. 2014

probes = {(r["hook"], r["task"]): r for r in csv.DictReader(open(out / "probe_results.csv"))}
behavior = {(r["task"], r["variant"]): 100 * float(r["acc_norm"]) for r in csv.DictReader(open(out / "behavioral_results.csv"))}

fig = S.figure(2.3)
ax = fig.add_subplot()
x = list(range(len(HOOKS)))
for task, color in TASKS.items():
    m = [100 * float(probes[h, task]["acc_norm"]) for h in HOOKS]
    s = [100 * float(probes[h, task]["std"]) for h in HOOKS]
    ax.plot(x, m, color=color, marker="o", label=S.tex(f"{WORDS[task]}, {WORDS['probe']}"))
    ax.fill_between(x, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)], color=color, alpha=0.18, lw=0)
for task, color in TASKS.items():
    ax.axhline(behavior[task, "letter"], color=color, ls=S.DASH, lw=1.1, label=S.tex(f"{WORDS[task]}, {WORDS['answer']}"))
ax.axhline(HUMAN_EMOTION, color=S.INK2, ls=S.DOT, lw=0.9, label=S.tex(WORDS["human"]))

pct_text = lambda v: f"{v:.0f} %" if lang == "fr" else f"{v:.0f}%"  # S.label escapes it for LaTeX

# probe-behavior gap on emotion, at the last layer: values only, the legend names the series
top, bottom = 100 * float(probes["L29", "emotion"]["acc_norm"]), behavior["emotion", "letter"]
ax.annotate("", xy=(x[-1] - 0.35, bottom + 1), xytext=(x[-1] - 0.35, top - 1),
            arrowprops=dict(arrowstyle="<->", color=S.INK, lw=0.7, shrinkA=0, shrinkB=0))
S.label(ax, x[-1] - 0.5, (top + HUMAN_EMOTION) / 2, f"{top - bottom:.0f} pts", ha="right", va="center")
S.label(ax, x[-1] + 0.15, top + 3, pct_text(top), ha="center", va="bottom", color=S.ORANGE)
S.label(ax, x[0], bottom + 2.5, pct_text(bottom), ha="left", va="bottom", color=S.ORANGE)

ax.set_xticks(x, [WORDS["enc"], WORDS["adapt"], *[h[1:] for h in HOOKS[2:]]])
ax.set_xlim(-0.4, x[-1] + 0.4)
ax.set_ylim(-5, 105)
ax.yaxis.set_major_formatter(S.pct(0))
ax.set_xlabel(S.tex(WORDS["x"]))
ax.set_ylabel(S.tex(WORDS["y"]))
S.legend_out(ax)
S.save(fig, "main_figure", out_dir="figures")
