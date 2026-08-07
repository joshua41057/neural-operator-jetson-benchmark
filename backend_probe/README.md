# Backend admission probe (§7, Appendix G)

The precision axis holds the backend fixed. This probe does the complement: the backend
is varied while precision is held at FP32, so the admission criterion of Eq. (4) is
applied to the **deployment toolchain** rather than to the datatype. Four families,
TorchScript / `torch.compile` / ONNX→TensorRT, at Burgers r=2048 and Darcy 141×141.

**This probe uses a separate harness** — five-iteration warmup, twenty timed iterations in
a single run — and **does not assert the clock-pinning preflight** of `harness/`. Its
latencies are therefore read only as *within-run ratios* and are never placed on the
timing axis used elsewhere in the paper. Run-to-run dispersion is correspondingly wider
(coefficient of variation 1.6–19.0% across the eight eager baselines).

Outcome: only the dense branch–trunk path reaches TensorRT. The recorded rejection
boundary for each of the other three is given in Table 48.
