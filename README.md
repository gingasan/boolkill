# Empower Nested Boolean Logic via Self-Supervised Curriculum Learning

This repo is for the EMNLP 2023 conference paper [Empower Nested Boolean Logic via Self-Supervised Curriculum Learning](https://arxiv.org/abs/2310.05450).



## BoolKill

*BoolKill* is a new benchmark that measures language models' boolean logic capability. (u01 -> $u_{01}$, a58 -> $\tilde{u}_{58}$)

To generate additional boolkill-like data, please refer to `data/gendata.ipynb`.



## Experiment

There are two main experiments in our paper.

**Nested Boolean Logic**

```bash
bash clr.sh
```



**Boolean to Complex Logic**

From a high-level perspective, boolean logic serves as an early curriculum for complex logic in logical end tasks.

```bash
bash mrc.sh
```

Results on dev sets:

| *DeBERTa-V3-base*                      | ReClor | DREAM | LogiQA |
| :------------------------------------- | ------ | ----- | ------ |
| sp                                     | 57.4   | 75.5  | 51.9   |
| $u_{01}$ -> sp                         | 62.0   | 79.4  | 54.9   |
| $u_{01}$ -> $u_{02}$ -> sp             | 63.0   | 79.2  | 55.1   |
| $u_{01}$ -> $u_{02}$ -> $u_{03}$ -> sp | 61.6   | 80.0  | 55.8   |

