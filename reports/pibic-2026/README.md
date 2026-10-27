# PIBIC 2025/2026 final report

This folder holds the final activity report of the PIBIC undergraduate research grant (ITA and CNPq) that funded this project from 1 September 2025 to 31 August 2026. The student was Victor Elias Vicentin and the supervisor Prof. Dr. Franciole da Cunha Marinho.

| Folder | Version | Title | Pages |
| --- | --- | --- | --- |
| [`pt/`](pt/) | Portuguese original, on the PIBIC-ITA template | Estudos de desempenho do sistema de fótons APE | 24 |
| [`en/`](en/) | English translation | Performance Studies of the APE Photon System | 25 |

The title is the one under which the grant project was registered in the ITA-PIBIC 2025/2026 results. Both copies are the version dated 14 October 2026. The English text follows the Portuguese paragraph by paragraph and keeps its numbers, figures, tables and references. Where the two differ, the Portuguese is the reference.

## Contents

The report describes the simulated deposits in one octant of the DUNE Far Detector 3, with the photon counts rescaled from 100 MeV to 10 MeV, the graph built from the active photon-detector channels and the training protocol. It then compares eleven message-passing encoders of matched capacity, trained and evaluated in the run of 8 August 2026, and sets the best of them against gradient-boosted trees (XGBoost and LightGBM) on the same test events, the dense network of the previous stage (2024), a calibrated light barycentre and the published APEX resolution. The last sections look at where the error concentrates, what limits the resolution, the difficulties met and the changes to the initial plan.

## Building

```bash
cd reports/pibic-2026/pt    # or en
tectonic -X compile main.tex
```

Tectonic is built on XeTeX, which the template needs for its TeX Gyre Heros body font, and it downloads the LaTeX packages it is missing on the first run. The Portuguese version cites in the ABNT author-date style of the template (`abntex2cite`), the English one with `natbib` and `abbrvnat`. The rendered PDFs are not tracked, since the repository keeps sources only. In `pt/`, `tectonic -X compile main_editavel.tex` builds the same report with fillable form fields in the two boxes of the supervisor's approval form and in the supervisor's telephone row, so the form can be completed in a PDF reader.

The nine figure PDFs in `figures/` and the two tables in `tables/` were generated from the run records of this repository by the report's build scripts, which are kept with the author's notes outside the repository; they are committed here as generated. The TikZ diagram `figures/arquitetura.tex` is written by hand. File names and LaTeX labels keep their Portuguese form in both versions.
