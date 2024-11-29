from sklearn.neighbors import kneighbors_graph
import numpy as np
import pandas as pd
import math
import datetime
import os
import shutil
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset


# Hyperparameters
InputFolderName = "/home/owkin/project/cytocommunity_results/fractions/raw/"
neighbors_order = 1
KNN_K = 3*neighbors_order*(neighbors_order+1)


## Import image name list.
Region_filename = InputFolderName + "ImageNameList.txt"
region_name_list = pd.read_csv(
        Region_filename,
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["Image"],  # set our own names for the columns
    )


## Below is for generation of topology structures (edges) of cellular spatial graphs.
ThisStep_OutputFolderName = f"/home/owkin/project/cytocommunity_results/fractions/graphs/neighbors_order_{neighbors_order}/"
if os.path.exists(ThisStep_OutputFolderName):
    shutil.rmtree(ThisStep_OutputFolderName)
os.makedirs(ThisStep_OutputFolderName)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("Constructing topology structures of KNN graphs...")
for graph_index in range(0, len(region_name_list)):

    print(f"This is image-{graph_index}")
    # Import target graph x/y coordinates.
    region_name = region_name_list.Image[graph_index]
    GraphCoord_filename = InputFolderName + region_name + "_Coordinates.txt"
    x_y_coordinates = np.loadtxt(GraphCoord_filename, dtype='float', delimiter="\t")

    K = KNN_K
    KNNgraph_sparse = kneighbors_graph(x_y_coordinates, K, mode='connectivity', include_self=False, n_jobs=-1)  #should NOT include itself as a nearest neighbor. Checked. "-1" means using all available cores.
    KNNgraph_AdjMat = KNNgraph_sparse.toarray()
    # Make the graph to undirected.
    KNNgraph_AdjMat_fix = KNNgraph_AdjMat + KNNgraph_AdjMat.T  #2min and cost one hundred memory.
    # Extract and write the edge index of the undirected graph.
    KNNgraph_EdgeIndex = np.argwhere(KNNgraph_AdjMat_fix > 0)  #1min
    filename0 = ThisStep_OutputFolderName + region_name + "_EdgeIndex.txt"
    np.savetxt(filename0, KNNgraph_EdgeIndex, delimiter='\t', fmt='%i')  #save as integers. Checked the bidirectional edges.
    
print("All topology structures have been generated!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# generate a node attribute matrix for each image.
num_nodes = []
for graph_index in range(0, len(region_name_list)):

    print(f"This is image-{graph_index}")
    region_name = region_name_list.Image[graph_index]
    node_attr_matrix = np.load(InputFolderName + region_name + "_DeconvolutionFrac.npy")
    print(node_attr_matrix)
    num_nodes.append(node_attr_matrix.shape[0])
    
    filename1 = ThisStep_OutputFolderName + region_name + "_NodeAttr.txt"
    np.savetxt(filename1, node_attr_matrix, delimiter='\t', fmt='%f')  #save as floats.
    

max_nodes = math.ceil(max(num_nodes))  # generate the max number of cells and store this value to .txt for the next step.
MaxNumNodes_filename = ThisStep_OutputFolderName + "MaxNumNodes.txt"
print("saving file "+MaxNumNodes_filename)
with open(MaxNumNodes_filename, 'w') as fp1:
    fp1.write("%i\n" % max_nodes)

print("All node attribute matrices have been generated!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


## Below is for transforming input graphs into the data structure required by deep geometric learning. 
print("Start graph data structure transformation...")
# Construct ordinary Python list to hold all input graphs.
data_list = []
for i in range(0, len(region_name_list)):
    region_name = region_name_list.Image[i]

    # Import network topology.
    EdgeIndex_filename = ThisStep_OutputFolderName + region_name + "_EdgeIndex.txt"
    edge_ndarray = np.loadtxt(EdgeIndex_filename, dtype = 'int64', delimiter = "\t")
    edge_index = torch.from_numpy(edge_ndarray)
    #print(edge_index.type()) #should be torch.LongTensor due to its dtype=torch.int64

    # Import node attribute.
    NodeAttr_filename = ThisStep_OutputFolderName + region_name + "_NodeAttr.txt"
    x_ndarray = np.loadtxt(NodeAttr_filename, dtype='float32', delimiter="\t")  #should be float32 not float or float64.
    x = torch.from_numpy(x_ndarray)
    #print(x.type()) #should be torch.FloatTensor not torch.DoubleTensor.
    
    data = Data(x=x, edge_index=edge_index.t().contiguous())
    data_list.append(data)

# Define "SpatialOmicsImageDataset" class based on ordinary Python list.
class SpatialOmicsImageDataset(InMemoryDataset):                                         
    def __init__(self, root, transform=None, pre_transform=None):
        super(SpatialOmicsImageDataset, self).__init__(root, transform, pre_transform)  
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SpatialOmicsImageDataset.pt']                                           

    def download(self):
        pass
    
    def process(self):
        # Read data_list into huge `Data` list.
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Create an object of this "SpatialOmicsImageDataset" class.
dataset = SpatialOmicsImageDataset(ThisStep_OutputFolderName, transform=T.ToDense(max_nodes))
print("Step1 done!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



