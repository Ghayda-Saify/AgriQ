"""Build both authorship variants of the paper from one source.

NGIA-QC 2026 reviews double-blind, so the submitted PDF must carry no author
names, while the supervisor and the university want a named copy:

    main.pdf            named    -- supervisor, university, your own records
    main_anonymous.pdf  blinded  -- the file to submit

Keeping two copies of main.tex would guarantee they drift apart, so the author
block is switched by the \\ifblind toggle and both variants are built here from
the same source every run.

Two failures in an earlier version of this script are worth remembering,
because both reported success while producing a named "anonymous" PDF:

  * It rewrote the first match of the toggle command in the file, which turned
    out to be an occurrence inside a comment. The toggle itself never changed.
    Now only the line carrying the AUTHORSHIP-TOGGLE marker is rewritten, and
    the script refuses to continue unless exactly one such line exists.
  * It then "verified" the result by searching the PDF's raw bytes. PDF text
    lives in compressed streams, so that search found nothing in either file
    and passed vacuously. Verification now extracts real text with pypdf.
"""
import io
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER = os.path.join(ROOT, "paper")
TEX = os.path.join(PAPER, "main.tex")
LATEX_BIN = r"C:\Users\ASUS\AppData\Roaming\TinyTeX\bin\windows"
MARKER = "AUTHORSHIP-TOGGLE"

# Must never appear anywhere in the blinded PDF's extracted text.
IDENTIFIERS = ["Ghaydaa", "Saify", "Tala", "Alhendi", "Najah", "AgriQ"]

try:
    import pypdf
except ImportError:
    sys.exit("pypdf is required to verify the blinded PDF: pip install pypdf")


def run(cmd):
    env = dict(os.environ)
    env["PATH"] = env.get("PATH", "") + os.pathsep + LATEX_BIN
    return subprocess.run(cmd, cwd=PAPER, env=env, shell=True,
                          capture_output=True, text=True)


def set_blind(flag):
    """Rewrite only the marked toggle line."""
    lines = io.open(TEX, encoding="utf-8", newline="").read().split("\n")
    hits = [i for i, ln in enumerate(lines) if MARKER in ln]
    if len(hits) != 1:
        sys.exit(f"expected exactly one {MARKER} line in main.tex, found {len(hits)}")
    i = hits[0]
    cmd = r"\blindtrue" if flag else r"\blindfalse"
    lines[i] = re.sub(r"\\blind(true|false)", cmd.replace("\\", "\\\\"), lines[i])
    if cmd not in lines[i]:
        sys.exit(f"could not set {cmd} on the toggle line: {lines[i]!r}")
    io.open(TEX, "w", encoding="utf-8", newline="").write("\n".join(lines))


def build(label):
    for step in ("pdflatex -interaction=nonstopmode main",
                 "bibtex main",
                 "pdflatex -interaction=nonstopmode main",
                 "pdflatex -interaction=nonstopmode main"):
        run(step)
    log = io.open(os.path.join(PAPER, "main.log"), encoding="utf-8",
                  errors="ignore").read()
    pages = re.findall(r"\((\d+) pages", log)
    errs = len(re.findall(r"^!", log, re.M))
    print(f"  {label:<10} {pages[-1] if pages else '?':>2} pages | errors {errs} | "
          f"overfull {log.count('Overfull')} | "
          f"undefined {len(re.findall(r'undefined', log, re.I))}")
    if errs:
        sys.exit(f"{label} build reported {errs} LaTeX errors")
    return os.path.join(PAPER, "main.pdf")


def pdf_text(path):
    return "\n".join((p.extract_text() or "") for p in pypdf.PdfReader(path).pages)


print("building both authorship variants from one source\n")

set_blind(True)
blind_out = os.path.join(PAPER, "main_anonymous.pdf")
shutil.copy(build("anonymous"), blind_out)

leaked = sorted({w for w in IDENTIFIERS if w.lower() in pdf_text(blind_out).lower()})
if leaked:
    os.remove(blind_out)
    sys.exit(f"REFUSING: blinded PDF contained {leaked}; it has been deleted")
print(f"  verified   none of {IDENTIFIERS} appear in the blinded PDF text")

set_blind(False)
named = build("named")
present = sorted({w for w in ("Ghaydaa", "Saify", "Tala", "Alhendi")
                  if w.lower() in pdf_text(named).lower()})
if len(present) < 4:
    sys.exit(f"named PDF is missing author names, found only {present}")
print(f"  verified   named PDF carries {present}")

print("\n  main.pdf            named    -> supervisor and university")
print("  main_anonymous.pdf  blinded  -> submit this one")
