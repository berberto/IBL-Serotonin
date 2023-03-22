# GLM-HMM analysis of IBL behaviour data

Code adapted from [zashwood/glm-hmm](https://github.com/zashwood/glm-hmm) and [guidomeijer/IBL-Serotonin](https://github.com/guidomeijer/IBL-Serotonin/tree/master/Behavior/GLM-HMM).

## Setup

### `iblenv` environment

First, create and setup the `iblenv` conda environment, following the [instructions here](https://github.com/int-brain-lab/iblenv).
```bash
cd /path/of/choice
conda create --name iblenv python=3.9 --yes
conda activate iblenv
git clone https://github.com/int-brain-lab/iblapps
pip install --editable iblapps
git clone https://github.com/int-brain-lab/iblenv
cd iblenv
pip install --requirement requirements.txt
```

### Install the `ssm` package

We use version 0.0.1 of the Bayesian State Space Modeling framework from Scott Linderman's lab to perform GLM-HMM inference. Within the `iblenv` environment, install the forked version of the `ssm` package available [here](https://github.com/zashwood/ssm).  This is a lightly modified version of the master branch of the ssm package available at [https://github.com/lindermanlab/ssm](https://github.com/lindermanlab/ssm). It is modified so as to handle violation trials as described in Section 4 of the manuscript. 
    
```bash
conda activate iblenv
cd /path/of/choice
git clone https://github.com/zashwood/ssm
cd ssm
pip install numpy cython
pip install -e .
```

## Usage

```bash
git clone https://github.com/int-brain-lab/GLM-HMM.git
cd GLM-HMM
conda activate iblenv
python 1_create_design_mat.py
# etc
```
