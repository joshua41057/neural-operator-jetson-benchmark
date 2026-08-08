# The four falsifiable predictions

Section 1 frames this work as **a hypothesis test rather than a leaderboard**. The
hypothesis is that deployed cost, precision portability and stress-boundary behavior are
organized by the *dominant computational mechanism of the executed path*, and not by
parameter count, sparsity structure, or architectural class.

It is falsifiable in four specific ways. Each is listed here with where it is tested and
what the measurement returned — including the one that does **not** transfer.

| # | Prediction | Tested in | Records | Outcome |
|---|---|---|---|---|
| **1** | Wide cost dispersion at matched parameter budgets, and within a fixed mechanism a cost elasticity to parameter count **below unity** | §5.1, §5.3 | `families/*/results/` short-run sweeps | **Holds.** At 234–242k on Burgers r=2048 median latency spans 3.2–48.1 ms. Endpoint log–log elasticity Small→Large stays in 0.07–0.78 for all four Darcy sweeps. |
| **2** | The mechanism-level cost partition persists when the same checkpoints are reinstantiated on a server-class accelerator | §8.2 | `gh200/` — 27 cases x 3 repetitions | **Holds with a qualification.** The two-way partition persists on both workloads; the full four-way ordering persists on one of the two. |
| **3** | Removing a computational branch moves deployed cost **more** than reducing depth and learned capacity does | §8.1 (near-asset), §8.2 (server-reference) | `families/heat_exchanger/` | **Holds on the near-asset target and reverses on the server-reference platform**, where the depth-reduction arm overtakes the branch-removal arm. This is the prediction that does not transfer, and the paper reports it as such. |
| **4** | Reachability of a production inference runtime follows the same mechanism separation as datatype support, not the architectural label | §7, Appendix G | `backend_probe/` | **Holds for the two-way separation.** The ranking *inside* the restricted group does not transfer: FNO and WNO exchange places between the precision axis and the toolchain axis. |

## Why prediction 3 is reported even though it fails to transfer

A hypothesis test is only informative if a negative result is reported with the same
weight as a positive one. Prediction 3 holds on the platform the paper is about and
reverses on the reference platform; the paper states this in §1, §8.2 and §12 rather
than restricting the claim to the platform where it survives.

## Status

These are pointers into the records, not re-analysis scripts. The tabulated values behind
every row are checked by `verification/verify_all.py`.
