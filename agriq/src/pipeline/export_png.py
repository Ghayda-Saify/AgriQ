"""Regenerate the three paper figures as PNG for the Word export.

Reuses make_figures.py unchanged by monkey-patching the savefig extension,
so the Word figures are guaranteed identical to the PDF ones.
"""
import os
import sys

ROOT = r"g:\Uni\AgriQ An AI and Quantum-Optimized System for Smart Crop Planning\agriq"
sys.path.insert(0, os.path.join(ROOT, "src", "pipeline"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

_orig = matplotlib.figure.Figure.savefig


def savefig_png(self, fname, *a, **kw):
    if isinstance(fname, str) and fname.endswith(".pdf"):
        fname = fname[:-4] + ".png"
        kw.setdefault("dpi", 300)
    return _orig(self, fname, *a, **kw)


matplotlib.figure.Figure.savefig = savefig_png

import make_figures  # noqa: E402

make_figures.fig_encoding()
make_figures.fig_map()
make_figures.fig_water()

FIG = os.path.join(ROOT, "paper", "figures")
for f in sorted(os.listdir(FIG)):
    if f.endswith(".png"):
        print(" ", f, os.path.getsize(os.path.join(FIG, f)) // 1024, "KB")
