# ML4Graphs

## Overview

This project implements and extends the paper [Question Answering by Reasoning Across Documents
 with Graph Convolutional Networks](https://arxiv.org/pdf/1808.09920) by de Cao et al. 

## Getting Started

To get started with this project:

1. **Setup the environment**:
```
conda create -n ML4Graphs python=3.9
conda activate ML4G

git  clone https://github.com/LarsdeWolf/ML4G.git
cd ML4G
pip install -r requirements.txt
```

2. **Construct Entity Graphs**:
```
python graph.py
```
Note: this can take a while depending on the available hardware. 

3. **Run Experiments**:
```
python experiments.py
```
