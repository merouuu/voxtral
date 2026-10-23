"""Functional diagram of Voxtral (article style): Whisper encoder -> adapter -> language decoder,
with the probe points of the main figure marked on a "probe point" axis (same labels as main_figure).

Architecture checked against the Voxtral paper (arXiv 2507.13264, §2) and the Hugging Face implementation
(transformers/models/voxtral): log-Mel 128 bins, hop 160 (100 Hz); Whisper large-v3 encoder = conv stem
(2 conv1d, stride 2) + 32 bidirectional self-attention layers, 1280-d, 50 Hz, 30-s chunks; adapter = 4 consecutive
frames concatenated (5120-d) -> linear -> GELU -> linear -> 3072-d, 12.5 Hz; decoder = Ministral 3B,
30 layers (RMSNorm, causal self-attention, RMSNorm, SwiGLU FFN), 3072-d, LM head.
Probe hooks (extract.py): audio_tower output ("enc."), multi_modal_projector output ("ad."),
language_model.layers[l] output at the audio positions ("0" ... "29"); mean over the audio frames.

Usage: python plot_architecture.py [en|fr] [--no-tex]   (English by default -> figures/architecture_en.*)
"""
import sys
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import article_style as S

lang = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "en"
S.setup_style(lang=lang, usetex="--no-tex" not in sys.argv)

LAYERS, N_DEC = [0, 4, 8, 12, 16, 20, 24, 29], 30
W = {
    "en": {"audio": "audio", "mel": "log-Mel\n128 bins", "hz12": "12.5 Hz", "enc": "Whisper encoder (large-v3, 640M)",
           "conv": "conv stem\n(2x down)", "encl": "32 layers\nbidirectional self-attention\nMLP (GELU)",
           "enc_out": "50 Hz, 1280-d", "ad": "adapter (25M)", "cat": "concat\n4 frames", "mlp": "MLP\n(GELU)",
           "ad_out": "12.5 Hz, 3072-d", "ad_note": "4x downsampling", "seq": "input sequence", "prompt": "prompt", "audio_pos": "audio", "question": "question",
           "text_in": "text (prompt, question)", "tokenizer": "tokenizer", "embed": "vector table\n(400M)", "dec": "language decoder (Ministral 3B)",
           "layer": "each layer: causal attention + FFN",
           "dec_layers": "30 layers, 3072-d", "lm": "LM\nhead", "text_out": "text",
           "axis": "probe point (encoder, adapter, decoder layer)", "enc_t": "enc.", "ad_t": "ad.",
           "leg_probe": "probe point: linear classifier on the mean over the audio positions",
           "leg_audio": "audio positions (what the probe reads)", "leg_text": "text tokens"},
    "fr": {"audio": "audio", "mel": "log-Mel\n128 bandes", "hz12": "12,5 Hz", "enc": "encodeur Whisper (large-v3, 640M)",
           "conv": "conv. d'entrée\n(sous-éch. 2x)", "encl": "32 couches\nself-attention bidir.\nMLP (GELU)",
           "enc_out": "50 Hz, 1280 dim", "ad": "adaptateur (25M)", "cat": "concat.\n4 trames", "mlp": "MLP\n(GELU)",
           "ad_out": "12,5 Hz, 3072 dim", "ad_note": "sous-éch. 4x", "seq": "séquence d'entrée", "prompt": "prompt", "audio_pos": "audio", "question": "question",
           "text_in": "texte (prompt, question)", "tokenizer": "tokenizer", "embed": "table de vecteurs\n(400M)", "dec": "décodeur de langage (Ministral 3B)",
           "layer": "chaque couche : attention causale + FFN",
           "dec_layers": "30 couches, 3072 dim", "lm": "tête\nLM", "text_out": "texte",
           "axis": "point de sonde (encodeur, adaptateur, couche du décodeur)", "enc_t": "enc.", "ad_t": "ad.",
           "leg_probe": "point de sonde : classifieur linéaire sur la moyenne des positions audio",
           "leg_audio": "positions audio (ce que lit la sonde)", "leg_text": "tokens texte"},
}[lang]
T = S.tex
DIM = (lambda n: f"{n}-d") if lang == "en" else (lambda n: f"{n} dim")
SMALL, TINY = S.FS - 1.5, S.FS - 2.5
FILL, FILL2 = "#f4f4f1", "#e6e5e0"

fig = S.figure(3.5, width=10.45)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
Y = 24  # main data path


def box(x0, x1, y0, y1, text=None, fill="white", lw=0.7, fs=SMALL, title=None, ec=S.INK):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.6",
                                fc=fill, ec=ec, lw=lw, zorder=2))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, T(text), ha="center", va="center", fontsize=fs, color=S.INK,
                zorder=3, linespacing=1.15)
    if title:
        ax.text(x0 + 0.9, y1 - 0.8, T(title), ha="left", va="top", fontsize=SMALL, color=S.INK, zorder=3)


def arrow(x0, x1, y0=Y, y1=Y, rate=None, dim=None, **kw):
    """An arrow along the data path; `rate` (frame rate) is written above it and `dim` (vector size) below,
    only where the value changes."""
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.7, color=S.INK,
                                 shrinkA=0, shrinkB=0, zorder=4, **kw))
    if rate:
        note((x0 + x1) / 2, Y + 1.1, rate)
    if dim:
        note((x0 + x1) / 2 + (1.2 if dim == 131072 else 0), Y - 1.1, DIM(dim), va="top")


def probe(x, y=Y):
    ax.plot([x], [y], "o", color=S.INK, ms=4.5, zorder=6)
    return x


def note(x, y, s, va="bottom"):
    ax.text(x, y, T(s), ha="center", va=va, fontsize=TINY, color=S.INK2, zorder=5)


# ---- input: waveform + log-Mel -------------------------------------------------------------------------
t = np.linspace(0, 1, 200)
ax.plot(0.4 + 3.0 * t, Y + 2.2 * np.sin(31 * t) * np.exp(-((t - 0.5) ** 2) / 0.08), color=S.INK, lw=0.6, zorder=3)
ax.text(1.9, Y - 3.4, T(W["audio"]), ha="center", va="top", fontsize=SMALL)
arrow(3.6, 8.4, rate="16 kHz")
box(8.4, 15.2, Y - 3.2, Y + 3.2, W["mel"])
arrow(15.2, 21.8, rate="100 Hz", dim=128)

# ---- Whisper encoder -----------------------------------------------------------------------------------
E0, E1 = 21.8, 56
box(E0, E1, 14, 34, fill=FILL, title=W["enc"])
box(23.2, 31.6, Y - 3.8, Y + 3.8, W["conv"], fs=TINY)
arrow(31.6, 38.2, rate="50 Hz", dim=1280)
for d in (1.4, 0.7):  # stacked cards = repeated layers
    box(38.2 + d, 54.6 + d, Y - 4.8 - d, Y + 4.8 - d, fill="white", ec=S.LINE)
box(38.2, 54.6, Y - 4.8, Y + 4.8, W["encl"], fs=TINY)
arrow(E1, 66)
x_enc = probe(61)

# ---- adapter -------------------------------------------------------------------------------------------
A0, A1 = 66, 85.5
box(A0, A1, 14, 34, fill=FILL, title=W["ad"])
box(67.5, 73.5, Y - 3.8, Y + 3.8, W["cat"], fs=TINY)
arrow(73.5, 80.1, rate=W["hz12"], dim=5120)
box(80.1, 84.4, Y - 3.8, Y + 3.8, W["mlp"], fs=TINY)
note((A0 + A1) / 2, 16.3, W["ad_note"], va="center")
ax.plot([A1, 92], [Y, Y], color=S.INK, lw=0.7, zorder=4, solid_capstyle="butt")  # adapter output ...
x_ad = probe(89.5)
note(87.3, Y - 1.1, DIM(3072), va="top")

# ---- input sequence: prompt | audio | question ----------------------------------------------------------
S0, S1, cells = 93, 107, "ttaaaaatt"
cw = (S1 - S0) / len(cells)
groups = {"t0": (S0, S0 + 2 * cw), "a": (S0 + 2 * cw, S0 + 7 * cw), "t1": (S0 + 7 * cw, S1)}
xc = {k: (lo + hi) / 2 for k, (lo, hi) in groups.items()}
for i, c in enumerate(cells):
    ax.add_patch(Rectangle((S0 + i * cw, Y - 1.6), cw, 3.2, fc=S.INK2 if c == "a" else "white", ec=S.INK, lw=0.5, zorder=3))
for k, lab in (("t0", W["prompt"]), ("a", W["audio_pos"]), ("t1", W["question"])):
    note(xc[k], Y - 5.0, lab, va="top")
note((S0 + S1) / 2, Y - 7.2, W["seq"], va="top")
# ... goes under the prompt cells and up into the audio cells
ax.plot([92, 92, xc["a"]], [Y, Y - 3.4, Y - 3.4], color=S.INK, lw=0.7, zorder=4, solid_joinstyle="miter")
arrow(xc["a"], xc["a"], Y - 3.4, Y - 1.6)
# text input (prompt, question) comes from the top: a small "lines of text" glyph, tokenizer, vector table
xm = (S0 + S1) / 2
for i, wdt in enumerate((5.0, 3.6, 4.4)):
    ax.plot([xm - 2.5, xm - 2.5 + wdt], [42.6 - 1.1 * i, 42.6 - 1.1 * i], color=S.INK, lw=0.8, zorder=3, solid_capstyle="round")
ax.text(xm + 3.6, 41.5, T(W["text_in"]), ha="left", va="center", fontsize=SMALL)
arrow(xm, xm, 39.6, 38.0)
box(xm - 5.8, xm + 5.8, 34.8, 38.0, W["tokenizer"], fs=TINY)
arrow(xm, xm, 34.8, 33.2)
box(xm - 5.8, xm + 5.8, 29.6, 33.2, W["embed"], fs=TINY)
# the vector table feeds the prompt cells and the question cells
ax.plot([xm, xm], [29.6, 28], color=S.INK, lw=0.7, zorder=4)
ax.plot([xc["t0"], xc["t1"]], [28, 28], color=S.INK, lw=0.7, zorder=4, solid_capstyle="butt")
arrow(xc["t0"], xc["t0"], 28, Y + 1.6)
arrow(xc["t1"], xc["t1"], 28, Y + 1.6)
arrow(S1, 112.6)

# ---- language decoder ----------------------------------------------------------------------------------
D0, D1 = 112.6, 139.4
box(D0, D1, 14.5, 34, fill=FILL, title=W["dec"])
L0, L1, LY0, LY1 = 114, 131.5, Y - 5.5, Y + 5.5
pitch = (L1 - L0) / N_DEC
xs_layer = []
for k in range(N_DEC):
    probed = k in LAYERS
    x = L0 + k * pitch
    ax.add_patch(Rectangle((x, LY0), pitch * 0.72, LY1 - LY0, fc="white", ec=S.INK if probed else S.LINE,
                           lw=0.6 if probed else 0.5, zorder=3))
    xs_layer.append(x + pitch * 0.36)
    if probed:
        probe(xs_layer[-1], LY0 - 1.0)  # same marker as "enc." and "ad.": a probe point
note((L0 + L1) / 2, LY1 + 0.6, W["dec_layers"])
ax.text((D0 + D1) / 2, 34.8, T(W["layer"]), ha="center", va="bottom", fontsize=TINY, color=S.INK, linespacing=1.3)
arrow(L1, 133)
box(133, 137.6, Y - 3.2, Y + 3.2, W["lm"], fs=TINY)
arrow(137.6, 147.4, dim=131072)
ax.text(145.2, Y - 3.4, T(W["text_out"]), ha="center", va="top", fontsize=SMALL)

# ---- probe axis (same ticks as the main figure) --------------------------------------------------------
AY = 3.2
xs_probe = [x_enc, x_ad] + [xs_layer[k] for k in LAYERS]
labels = [W["enc_t"], W["ad_t"]] + [str(k) for k in LAYERS]
ax.plot([xs_probe[0] - 2, xs_probe[-1] + 2], [AY, AY], color=S.INK, lw=0.7, zorder=2, solid_capstyle="butt")
for x, lab, y_top in zip(xs_probe, labels, [Y, Y] + [LY0 - 1.0] * len(LAYERS)):
    ax.plot([x, x], [AY, y_top], color=S.INK2, ls=S.DOT, lw=0.6, zorder=5)  # drawn over the boxes
    ax.plot([x, x], [AY - 0.6, AY], color=S.INK, lw=0.7, zorder=2)
    ax.text(x, AY - 1.0, T(lab), ha="center", va="top", fontsize=S.FS - 1)
ax.text((xs_probe[0] + xs_probe[-1]) / 2, -1.2, T(W["axis"]), ha="center", va="top", fontsize=S.FS)
ax.set_xlim(0, 149)
ax.set_ylim(-4.5, 44)

handles = [Line2D([], [], color=S.INK, marker="o", ls="none", label=T(W["leg_probe"])),
           Patch(fc=S.INK2, ec=S.INK, lw=0.5, label=T(W["leg_audio"])),
           Patch(fc="white", ec=S.INK, lw=0.5, label=T(W["leg_text"]))]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=TINY, frameon=False,
          handlelength=1.2, borderaxespad=0.3)
S.save(fig, "architecture", out_dir="figures")
