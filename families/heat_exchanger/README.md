# Heat Exchanger — deployment measurements

## Contribution boundary

The irregular-mesh heat exchanger case is an **application-facing runtime path** for the
Sp²GNO family. Two studies meet here, and the split is deliberate.

**Contributed by the companion study** (Howes et al., *Graph Neural Operator Towards Edge
Deployability and Portability for Sparse-to-Dense, Real-Time Virtual Sensing on Irregular
Grids*) — https://github.com/joshua41057/virso-jetson-inference

- the graph-operator architecture (`sp2gno_model.py`)
- the training procedure and the trained checkpoints
- the ANSYS Fluent RANS dataset and its preprocessing

**Contributed by this work** — the files in this directory:

- the sustained and short-run deployment sweeps over the three pathway variants
  (full 10-layer, spectral-only, 2-layer)
- precision-mode admission under the whole-path policy of the main paper
- board-level `tegrastats` telemetry and the parsed energy records
- Nsight operation-class and per-kernel evidence

The checkpoints are used here **purely as admitted executed paths**; this work does not
modify their training procedure. Section 11 of the paper states the same boundary.

## How this directory was assembled

Only files that the companion repository does **not** track were copied, so no file is
published twice under two different attributions.
