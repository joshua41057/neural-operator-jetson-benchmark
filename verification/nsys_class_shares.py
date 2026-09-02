#!/usr/bin/env python3
"""Kernel-class shares from an Nsight Systems cuda_gpu_kern_sum export.

This is the taxonomy behind Fig. 5 and the four *_nsys_full appendix tables.
Shares are the sum of the printed Time (%) column per class, so they reproduce
the tables directly rather than being recomputed from nanoseconds.

Conventions that are not obvious from the class names and are fixed here so
they cannot drift:
  * cublasLt::splitKreduce_kernel -> Reduction/Norm in every case, including
    the heat-exchanger ones where it carries 1.2% of the trace.
  * cuFFT preprocess_kernel / postprocess_kernel / kernel_wrapper -> Other.
  * RowwiseMomentsCUDAKernel -> Other; LayerNormForwardCUDAKernel and
    vectorized_layer_norm_kernel -> Reduction/Norm.
  * elementwise_kernel_with_index (linspace/arange) and upsample_* -> Other.
  * cudnn::* is Conv/Wavelet, including engines_precompiled::scalePackedTensor.

Nsight Systems truncates kernel names at ~88 characters, so a small number of
elementwise_kernel<...> rows that are movement functors cannot be distinguished
from activation functors in the text export; shares reproduce the published
values to within 0.5 percentage points for that reason.

    python3 verification/nsys_class_shares.py <stats.txt> [...]
"""
import re, sys, os

KERN = re.compile(r'\s*(\d+\.\d+)\s+(\d+)\s+(\d+)\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+\s+[\d.]+\s+(.*)$')
ORDER = ["FFT/Spectral", "Conv/Wavelet", "Dense/GEMM", "Activation/Elementwise",
         "Movement/Materialization", "Reduction/Norm", "Other"]


def klass(n):
    if 'splitKreduce' in n:                                    return 'Reduction/Norm'
    if any(k in n for k in ('preprocess_kernel', 'postprocess_kernel',
                            'kernel_wrapper', 'RowwiseMoments',
                            'elementwise_kernel_with_index', 'upsample')):
        return 'Other'
    if any(k in n for k in ('LayerNorm', 'layer_norm', 'reduce_kernel',
                            'batch_norm', 'softmax')):          return 'Reduction/Norm'
    if any(k in n for k in ('cudnn', 'convolve', 'conv_depthwise',
                            'dgrad', 'wgrad')):                 return 'Conv/Wavelet'
    if 'fft' in n.lower() or 'spRadix' in n:                    return 'FFT/Spectral'
    if any(k in n for k in ('sgemm', 'gemm', 'gemv', 'cutlass', 'trsm')):
        return 'Dense/GEMM'
    if any(k in n for k in ('CatArrayBatchedCopy', 'indexSelect', 'gpu_index_kernel',
                            'scatter_gather', 'index_elementwise_kernel',
                            'copy_', 'CopyKernel', 'permute', 'transpose')):
        return 'Movement/Materialization'
    return 'Activation/Elementwise'


def shares(path):
    txt = open(path, errors='replace').read()
    i = txt.find('cuda_gpu_kern_sum):')
    if i < 0:
        return None, 0.0
    seg = txt[i:]
    j = seg.find('\nProcessing [')
    if j > 0:
        seg = seg[:j]
    out, total_ns = {}, 0
    for line in seg.splitlines():
        m = KERN.match(line)
        if not m:
            continue
        pct, ns, name = float(m.group(1)), int(m.group(2)), m.group(4).strip()
        out[klass(name)] = out.get(klass(name), 0.0) + pct
        total_ns += ns
    return out, total_ns / 1e6


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print(f"{'case':44s} {'total ms':>9s} " + " ".join(f"{c.split('/')[0][:8]:>8s}" for c in ORDER))
    for p in sys.argv[1:]:
        sh, ms = shares(p)
        if sh is None:
            print(f"{os.path.basename(p):44s}  no cuda_gpu_kern_sum section")
            continue
        print(f"{os.path.basename(p)[:44]:44s} {ms:9.1f} " +
              " ".join(f"{sh.get(c, 0.0):8.1f}" for c in ORDER))
