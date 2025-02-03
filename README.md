# Machine Learning for Graphs

## Exploiting Centrality Information with Graph Convolutions for Network Representation Learning

This repository replicates part of a research paper. The code is based on the GraphSAGE algorithm developed by William Leif:  
[GraphSAGE GitHub](https://github.com/williamleif/GraphSAGE/blob/master/example_unsupervised.sh)

**GraphSAGE Extension**  
The core GraphSAGE codebase has been extended to implement a simple GraphCSC model for Degree Centrality and Bridge Strength.

---

## Requirements

- **Python 3.6.0** is required to run this code.
- After cloning the repository, update the directory paths in the following files:
  - `negativer_sampler.py` (lines 15 and 19)
  - `Extension_Paper.py`
  - `Replication_paper.py`
  - `utils.py`
  - `Prediictions_MAE_Paper_Extension.py`
  - `Predictions_MAE_Paper_replications.py`

---

## Usage

### Replicating Results

- The embeddings, test nodes, and train nodes are included in the repository.
- To replicate the results, simply run:
  - `Prediictions_MAE_Paper_Extension.py`
  - `Predictions_MAE_Paper_replications.py`
- These scripts do not require rerunning the entire model to recreate embeddings.

---

## Data

- Preprocessed data is located in the `data` folder.
- Preprocessed PPI data and centrality random walks are located in the `PPI_Data` folder.
