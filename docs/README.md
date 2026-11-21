# Laporan Tugas Besar 2

Build `main.pdf` from this folder:

```sh
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

(`latexmk -pdf main.tex` also works if Perl is installed.)

- `sections/`: one file per chapter, included from `main.tex`.
- `public/`: logos; drop `foto-kelompok.png` here to put the group photo on the cover.
- `public/figures/`: figures used in the report.
- `dafpus.bib`: references.
