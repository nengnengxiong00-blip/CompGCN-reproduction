# CompGCN Corr FFT Compatibility Fix

## Environment

- Dataset: FB15k-237
- Framework: PyTorch 1.13.1+cu117
- Model: CompGCN
- Operation: Circular Correlation (Corr)

## Problem

The original implementation used com_mult() with
torch.fft.rfft().

The old com_mult() function assumes that real and
imaginary components are stored in the last dimension.

However, torch.fft.rfft() returns a complex tensor.

Therefore, the implementation did not compute the
standard circular correlation correctly.

## Numerical Test

Input:
a = [1, 2, 3, 4]
b = [2, 1, 0, 3]

Before fix:
[19, 11, 11, 19]

Reference result:
[16, 18, 16, 10]

After fix:
[16, 18, 16, 10]

Numerical test: PASSED

## Gradient Test

Input shape: [4, 200]

Output shape: [4, 200]

Entity gradient finite: True
Relation gradient finite: True

Entity gradient norm: 19.275833129882812
Relation gradient norm: 20.21391487121582

Gradient test: PASSED

## Historical Results

The following historical Corr experiments were
completed before this fix:

- TransE + Corr
- DistMult + Corr
- ConvE + Corr

These results are preserved for reference.

The exact historical runtime code version still
requires verification.

The Corr experiments should be re-evaluated
using the corrected implementation.

## Remaining Work

- Verify the full CompGCN model forward pass.
- Verify the FFT implementation of cconv().
- Rerun the affected Corr baselines.
- Compare corrected results under the same
  evaluation protocol.
