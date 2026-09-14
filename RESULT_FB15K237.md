# FB15k-237 Reproduction Result

## Environment

GPU:
RTX 3080 Ti

PyTorch:
1.13.1+cu117

Dataset:
FB15k-237


## Training

Model:
CompGCN

Epoch:
Early stopping at 283


## Test Result

Tail MRR:
0.36975

Head MRR:
0.17087

Average MRR:
0.27031


## Compatibility Modification

helper.py modified:

torch.rfft -> torch.fft.rfft

torch.irfft -> torch.fft.irfft
