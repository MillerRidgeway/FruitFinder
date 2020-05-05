# Fickle Fruit Finder
Fruit image classification at scale using distributed PyTorch.

## Dataset
Designed for use with "Fruit Recognition Dataset", by Israr Hussain; Qianhua He; Zhuliang Chen; Wei Xie
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1310165.svg)](https://doi.org/10.5281/zenodo.1310165)

## Usage

Launch 4 node training session.

```bash
ipython dist_test.py 4
```
 
You can run from an interactive session allowing you to work with the model and metrics afterword

```python
from dist_test import *
model, metrics = dist_train(world_size, 0, auto=True)
```

Also from interactive session you can run multiple tests back to back. Results saved to 'metrics/metrics-all-timestamp.pt'

```python
from dist_test import *
test_mult([1,2,4,6])
```
