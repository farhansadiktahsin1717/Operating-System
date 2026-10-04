# Track 1 — Page Replacement (CSE-307 Operating Systems)

**Student:** Farhan Sadik Tahsin  **ID:** 202414072

A small, reproducible experiment comparing **FIFO**, **LRU**, **Optimal (Belady)**, a **learned eviction policy** (small decision tree) and a **Hybrid** (learned + LRU fallback) when the workload shifts.

## Scenario
A student's laptop with only 4 free memory frames and 10 pages. First half of the trace: the student works on one assignment (strong locality). Second half: the assignment is submitted and the student browses randomly (wide, bursty access).

## What the project studies
The same 800-reference trace (seed 307) is used for every policy. Page faults and hit ratios are reported separately for the locality phase and the shift phase. The learned policy is a depth-3 decision tree written from scratch. It is trained on a separate locality-only trace; training labels say whether Belady's oracle would evict a candidate page. At run time it only uses information available at the current fault (age, frequency, recency rank, page id). Results are also repeated on 20 extra traces.

## Requirements
- Ubuntu or another Unix-like system
- Python 3.10+
- No third-party Python packages (LaTeX only needed to rebuild the report)

## Run
```
./run.sh                      # or: python3 src/run_experiment.py
cd report && pdflatex term_paper.tex && pdflatex term_paper.tex
```
Outputs in `results/`:
- `access_trace.csv` — trace with phase labels
- `policy_metrics.csv` — faults and hit ratios per policy and phase
- `multiseed_summary.csv` — mean/std over 20 extra traces
- `rolling_hit.csv`, `hit_ratio.svg` — data and chart
- `learned_tree.txt` — the learned decision rules
- `summary.json` — compact summary

## Structure
```
src/run_experiment.py   implementation and experiment driver
results/                generated raw results and figures
report/term_paper.tex   LaTeX source of the report (PDF included)
run.sh                  one-command runner
```

## Main result (seed 307)
| Policy | Locality hit | Shift hit |
|---|---|---|
| FIFO | 75.25% | 58.00% |
| LRU | 75.75% | 57.50% |
| Optimal | 84.50% | 73.50% |
| Learned | 76.75% | 54.75% |
| Hybrid | 76.75% | 56.50% |

On the main trace the learned policy drops most, but over 20 traces LRU, Learned and Hybrid all drop about 20 points, so the differences are within noise.

## Note
Optimal uses future references by definition, so it is a benchmark, not an implementable online policy.

## AI assistance disclosure
An AI assistant (Claude) helped with implementation and drafting. I reviewed the code, results and analysis and can explain every part.
