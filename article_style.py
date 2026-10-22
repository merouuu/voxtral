"""Style des figures du mémoire, AUTONOME : matplotlib + numpy, rien d'autre.

Copie conforme du style de `make_article_figures.py` (même rcParams, mêmes
couleurs, mêmes aides), sans ses imports de croquis ni ses données : ce fichier
se donne tel quel à quelqu'un qui veut tracer « comme le mémoire ». Si le style
change là-bas, le recopier ici -- les deux ne se lisent pas l'un l'autre.

Le style (demandé le 6 oct. 2026, sur le modèle de la figure MAP(τ) de CLEWS) :
  - une colonne d'article, 85 mm, corps 8 pt ; texte composé par LaTeX
    (Latin Modern) ; sans LaTeX installé, `setup_style(usetex=False)` → STIX ;
  - cadre complet autour des axes, un marqueur par point mesuré ;
  - légende ENCADRÉE, à droite HORS du cadre (`legend_out`) -- jamais de nom
    écrit à côté d'un point ; sur le tracé, seulement des valeurs ;
  - une mesure par axe, jamais deux échelles y (empiler deux panneaux) ;
  - une couleur = un bras : bleu = métadonnées apprises, orange = audio appris,
    gris = règles / gelé / références, noir = décisions dures ;
  - pas de titre sur la figure : la légende du papier le porte.

Usage :

    import article_style as S
    S.setup_style(lang="fr")                 # ou lang="en", usetex=False
    fig = S.figure(2.2)
    ax = fig.add_subplot()
    ax.plot(x, y, marker="o", color=S.BLUE, label=S.tex("métadonnées"))
    S.legend_out(ax)
    S.save(fig, "ma_figure", out_dir=".")    # ma_figure.pdf + .png (ma_figure_en.* si lang="en")

    python article_style.py                  # trace demo_article_style.{pdf,png} ici (données inventées)
    python article_style.py --no-tex --lang en
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Une couleur = un sens.
INK, INK2, LINE = "#0b0b0b", "#52514e", "#c4c3bd"      # encre, gris (règles / gelé), filets (iso-risques)
BLUE, ORANGE, VIOLET, AQUA = "#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a"
DASH, DOT = (0, (4, 2.5)), (0, (1.2, 2))              # linestyle tireté, pointillé
FS = 8                                                 # corps de l'article
WIDTH = 3.35                                           # une colonne, 85 mm (la légende déborde à droite)
LANG = "fr"
USETEX = True

# Ce que LaTeX ne lit pas tel quel (les libellés s'écrivent en unicode, lisibles dans le code).
TEX = [("%", r"\%"), ("φ", r"$\phi$"), ("λ", r"$\lambda$"), ("≥", r"$\geq$"), ("−", r"$-$"), ("×", r"$\times$"),
       ("≈", r"$\approx$"), ("#", r"\#"), ("&", r"\&")]


def setup_style(lang="fr", usetex=True, width=None):
    """À appeler une fois avant de tracer. `lang` règle les nombres (0,1 % / 0.1%)
    et le suffixe `_en` de `save` ; `usetex=False` si LaTeX n'est pas installé."""
    global LANG, USETEX, WIDTH
    LANG, USETEX = lang, usetex
    if width:
        WIDTH = width
    plt.rcParams.update({
        "text.usetex": USETEX,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{lmodern}",
        "font.family": "serif", "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1,
        "legend.fontsize": FS - 1, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": 0.7,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
        "lines.linewidth": 1.4, "lines.markersize": 4.5, "lines.markeredgewidth": 0,
        "legend.frameon": True, "legend.fancybox": True, "legend.edgecolor": "0.8", "legend.framealpha": 1.0,
        "legend.handlelength": 2.0, "legend.borderaxespad": 0.0,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
    })


def tex(s):
    """Échappe un libellé unicode pour LaTeX (sans effet quand usetex=False)."""
    if not USETEX:
        return s
    for a, b in TEX:
        s = s.replace(a, b)
    return s


def num(v, nd=0):
    """Un pourcentage : 0,1 % en français, 0.1% en anglais."""
    t = f"{v:.{nd}f}"
    return tex(t.replace(".", ",") + " %" if LANG == "fr" else t + "%")


def thousands(x):
    """6 545 en français, 6,545 en anglais."""
    return f"{int(x):,}".replace(",", " ") if LANG == "fr" else f"{int(x):,}"


def pct(nd):
    """Formateur d'axe en pourcentage : `ax.xaxis.set_major_formatter(pct(1))`."""
    return FuncFormatter(lambda v, _pos: num(v, nd))


def figure(height, width=None):
    """Une figure d'une colonne (pouces) ; `width=7` pour deux colonnes."""
    return plt.figure(figsize=(width or WIDTH, height))


def label(ax, x, y, s, **kw):
    """Une VALEUR posée sur le tracé (jamais un nom de série : ça, c'est la légende)."""
    kw.setdefault("color", INK)
    kw.setdefault("fontsize", FS - 0.5)
    ax.text(x, y, tex(s), **kw)


def legend_out(ax, **kw):
    """La légende encadrée, à droite hors du cadre, alignée sur le haut des axes."""
    kw.setdefault("loc", "upper left")
    kw.setdefault("bbox_to_anchor", (1.03, 1.0))
    return ax.legend(**kw)


def iso_risk(ax, risks, scale, lam=10, at=None, va="bottom"):
    """Droites R = dup + λ·misat dans le plan (misat en x, dup en y), en gris fin.
    `scale` convertit un risque en unités d'axe (100 / n_requêtes pour des %) ;
    `at` = abscisses où écrire « R = … ». Poser les xlim AVANT d'appeler."""
    xs = np.array(ax.get_xlim())
    for i, R in enumerate(risks):
        ax.plot(xs, R * scale - lam * xs, color=LINE, lw=0.6, zorder=0)
        if at is not None:
            x = at[i]
            label(ax, x, R * scale - lam * x, f"R = {thousands(R)}", color=INK2, fontsize=FS - 1.5,
                  va=va, ha="left", clip_on=True)


def save(fig, name, out_dir="."):
    """`<name>.pdf` et `<name>.png` (suffixe `_en` en anglais), marges serrées."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name if LANG == "fr" else name + "_en"
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(out_dir / f"{name}.pdf")


def demo(out_dir="."):
    """Le plan (misattributions, duplications) avec trois bras -- DONNÉES INVENTÉES,
    seulement pour montrer le style."""
    n = 6545
    en = {"règle de Bayes, sans audio": "Bayes rule, no audio", "règle de Bayes, avec audio": "Bayes rule, with audio",
          "seuil unique": "single threshold",
          "misattributions (% des requêtes)": "misattributions (% of queries)",
          "duplications (% des requêtes)": "duplicates (% of queries)"}
    _ = lambda s: tex(en.get(s, s) if LANG == "en" else s)  # noqa: E731
    fig = figure(2.3)
    ax = fig.add_subplot()
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 4.5)
    iso_risk(ax, [100, 200], scale=100 / n, at=[0.02, 0.02])
    arms = [("règle de Bayes, sans audio", BLUE, "-", [(0.59, 0.41), (0.36, 0.92), (0.09, 1.68)]),
            ("règle de Bayes, avec audio", ORANGE, "-", [(0.45, 0.30), (0.20, 0.55), (0.05, 0.88)]),
            ("seuil unique", INK2, DASH, [(0.62, 0.70), (0.40, 1.40), (0.15, 2.60)])]
    for name, color, ls, pts in arms:
        x, y = np.array(pts).T
        ax.plot(x, y, color=color, ls=ls, marker="o", label=_(name))
    ax.plot([0.40], [3.9], "s", color=INK, label="Cluedo")
    ax.xaxis.set_major_formatter(pct(1))
    ax.yaxis.set_major_formatter(pct(1))            # autant de décimales que le pas des graduations
    ax.set_xlabel(_("misattributions (% des requêtes)"))
    ax.set_ylabel(_("duplications (% des requêtes)"))
    legend_out(ax)
    save(fig, "demo_article_style", out_dir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Trace une figure de démonstration du style (données inventées).")
    p.add_argument("--lang", choices=["fr", "en"], default="fr")
    p.add_argument("--no-tex", action="store_true", help="sans LaTeX : police STIX")
    p.add_argument("--out-dir", default=".")
    a = p.parse_args()
    setup_style(lang=a.lang, usetex=not a.no_tex)
    demo(a.out_dir)
