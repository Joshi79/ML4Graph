# Machine Learning for Graphs
## Exploiting Centrality Information with Graph Convolutions for Network Representation Learning

This repository is there for the reason to replicate part of the given Research paper. 
The code is based on the GraphSage algorithm which was created by Williams.
https://github.com/williamleif/GraphSAGE/blob/master/example_unsupervised.sh

The basis of the GraphSage code was extendend to be able to implement a simple GraphCSC model for Degree centrality and Bridge strength. 

It is important to Know to run the code in this repository, it is required to use ## Python 3.6.0. \
To be able to run the code, one needs to change also the direcotry path in the python files. 
Therefore after cloning the github one needs to change the direcoty in the files: 
Change the directory in line 15 and 19 in the file negativer_sampler.py
Furthermore in the Extension_Paper.py, Replication_paper.py, utils.py, Prediictions_MAE_Paper_Extension.py, and lastly Predictions_MAE_Paper_replications. \\

The embeddings and test and train nodes are also uploaded in the Github, therefore, one can replicate the results just by running  Prediictions_MAE_Paper_Extension.py, and lastly Predictions_MAE_Paper_replications without the need to rerun the entire model to create the embeddings. \\

The Preprocessed data can be found in the data folder. \
The preprocessed PPI data and the centrality random walks can be found in the PPI_Data foler. 
