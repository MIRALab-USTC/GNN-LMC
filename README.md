# LMC: Fast Training of GNNs via Subgraph Sampling with Provable Convergence

This is the code of paper 
**LMC: Fast Training of GNNs via Subgraph Sampling with Provable Convergence**. 
Zhihao Shi, Xize Liang, Jie Wang. ICLR 2023. [[arXiv](https://arxiv.org/abs/2302.00924)]
[[NeurIPS-Official](https://openreview.net/forum?id=5VBBA91N6n)]

## Dependencies
- Python 3.7
- PyTorch 1.9.0
- torch-geometric 1.7.2
- ogb 1.3.3
- hydra-core 1.1.0


## Reproduce the Results

### 1. Compile the subgraph sampling codes
To compile the subgraph sampling codes in the `csrc` directory, run the following commands.

```shell script
cd code
python setup.py
```

### 2. Reproduce the Results 
To reproduce the results,
please run the following commands.

```shell script
CUDA_VISIBLE_DEVICES=0 python main_large.py dataset=arxiv  model=gcn  model.json='[PATH of CODE]/json/gcn/arxiv/variant.json'
```

## Citation
If you find this code useful, please consider citing the following paper.
```
BibTeX Record
@inproceedings{
shi2023lmc,
title={{LMC}: Fast Training of {GNN}s via Subgraph Sampling with Provable Convergence},
author={Zhihao Shi and Xize Liang and Jie Wang},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=5VBBA91N6n}
}
```

## Acknowledgement
We refer to the code of [PyGAS](https://github.com/rusty1s/pyg_autoscale). Thanks for their contributions.






