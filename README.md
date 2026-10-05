# Research Portfolio — SEMTM0045

## Student Information

| Field | Details |
|---|---|
| **Name** | Thrijwal Krishnappa |
| **Student ID** | nw25230 |
| **Programme** | Data Science Methods and Practice |
| **Module** | SEMTM0045 |
| **Research Theme** | Theme 3 |
| **Research Topic** | Topic 3 |

## Project Title

**Beyond Accuracy: Evaluating Calibration Stability and Generalisability in Biomedical Predictive Models**

## Project Summary

This project develops a calibration-conscious validation framework for evaluating biomedical predictive models under cross-institutional dataset shift. Using MIMIC-III and eICU critical care datasets, the study compares Logistic Regression, XGBoost, and Neural Networks across discrimination, calibration, and clinical utility metrics, before and after post-hoc calibration correction.

---

## Repository Structure

```
research-portfolio-nw25230/
│
├── README.md                                         ← You are here
├── SEMTM0045_Theme3_Topic3_nw25230.pdf               ← Final submitted portfolio (PDF)
│
├── LaTeX/                                            ← All source files for the written report
│   ├── main.tex                                      ← Main LaTeX manuscript
│   ├── IEEEtran.cls                                  ← IEEE template class file
│   └── references.bib                                ← BibTeX bibliography file
│
├── Planning/                                         ← Project planning artefacts
│   ├── gantt_chart.pdf                               ← 12-week Gantt chart / milestone timeline
│   └── jira_link.txt                                 ← Link to Jira project board
│
└── Research_Evidence/                                ← Supporting research materials
    ├── literature_notes/                             ← Annotated notes on key papers
    ├── paper_summaries/                              ← CRAAP-evaluated paper summaries
    ├── research_gap_analysis.md                      ← Gap analysis working document
    └── methodology_sketches/                         ← Pipeline design notes
```

---

## Tools Used

| Tool | Purpose |
|---|---|
| **LaTeX (TeX Live 2025)** | Written report typesetting in IEEE format |
| **BibTeX / IEEEtran** | Reference management and citation formatting |
| **Jira** | Project planning, task tracking, and milestone management |
| **Python** | Data preprocessing, model development, and evaluation |
| **MIMIC-III** | Primary training dataset (critical care EHR) |
| **eICU Collaborative Research Database** | External validation dataset (multi-site ICU) |
| **GitHub** | Version control and audit trail |

---

## Project Planning

The full 12-week project timeline, task breakdown, and milestone tracking is managed in Jira:

🔗 **Jira Board:** [CAVICM Project Board](https://bristol-team-wnshveyf.atlassian.net/jira/software/projects/CAVICM/boards)

---

## Navigation Guide

- To read the full written portfolio, open `SEMTM0045_Theme3_Topic3_nw25230.pdf`
- To compile the LaTeX source, run `pdflatex main.tex` twice then `bibtex main` from the `LaTeX/` folder
- To view the project timeline visually, open `Planning/gantt_chart.pdf`
- Supporting evidence and literature analysis are in `Research_Evidence/`

---

## GitHub Repository

🔗 [https://github.com/UoB-DSMP2026/research-portfolio-nw25230](https://github.com/UoB-DSMP2026/research-portfolio-nw25230)
