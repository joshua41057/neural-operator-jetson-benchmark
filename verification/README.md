# Verification

Two independent checks. They run in opposite directions and are complementary.

| Script | Direction | What it catches |
|---|---|---|
| `verify_all.py` | table → record | a tabulated value that does not match the measured record |
| `prose_trace.py` | prose → table | a number quoted in the text that has no source in any table or figure |

Both need the paper's LaTeX sources:

```bash
PAPER_DIR=/path/to/overleaf_work python verify_all.py
PAPER_DIR=/path/to/overleaf_work python prose_trace.py
```

`agg_all.pkl` is the aggregate built from the records under `families/*/results/`.
`verify_all.py` re-derives each tabulated cell from it and compares against the value
printed in the `.tex` source.

Last run before submission: **1733 checks, 0 discrepancies**; **197 prose decimals and
33 percentages, 0 without a source**.
