Predictive Congestion Aware Routing using T-GCN
This project implements a Temporal Graph Convolutional Network (T-GCN) to predict traffic speed using the METR-LA dataset.
The model captures both spatial dependencies between road sensors and temporal traffic patterns to support congestion-aware routing and ETA estimation.

Python Environments and Packages required
python 3.10
torch
numpy
pandas
matplotlib
folium

In data folder:
METR-LA.h5 dataset
normalised adjacency matrix.pkl
distances_la_2012.csv for distances between sensors
graph_sensor_locations.csv for latitude and longitude coordinates of the sensors

Checkpoint folder has trained model weights and training loss history

models folder contains model architecture code
gcn_layer.py --> graph convolutional layer for spatial learning 
tgcn_model.py --> full TGCN model combining GCN and GRU

main.py loads pre processed data, initializes and trains the TGCN model, saves model weights and losses


notebooks folder contains:
data_preperation.ipynb --> loading and preprocessing of dataset
training_plots.ipynb --> it contains training plots along with eta and graph

Code execution:
1> notebooks/01_data_preperation.ipynb
2> Run notebooks/02_training_plots.ipynb for training plots and predictions using trained weights


