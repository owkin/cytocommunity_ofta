import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGraphConv, dense_mincut_pool
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
import os
import numpy as np
import pandas as pd
import datetime
import csv
import shutil
import yaml
from tqdm import tqdm


def tcn_learning_unsupervised(Image_Name, InputFolderName):

    device = torch.device("cuda:1")
    
    # Load hyperparameters from config file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    
    ## Hyperparameters
    graph = config["graph"]
    hyperparams = config["hyperparameters"]
    Num_TCN = hyperparams["Num_TCN"]
    Num_Run = hyperparams["Num_Run"]
    Num_Epoch = hyperparams["Num_Epoch"]
    Embedding_Dimension = hyperparams["Embedding_Dimension"]
    Learning_Rate = hyperparams["LearningRate"]
    
    
    Loss_Cutoff = 1e8
    
    
    ## Import image name list.
    Region_filename = InputFolderName + "ImageNameList.txt"
    region_name_list = pd.read_csv(
            Region_filename,
            sep="\t",  # tab-separated
            header=None,  # no heading row
            names=["Image"],  # set our own names for the columns
        )
    
    
    ## Load dataset from the constructed Dataset.
    LastStep_OutputFolderName = f"/home/owkin/project/cytocommunity_results/fractions/graphs/neighbors_order_{graph['neighbors_order']}/"
    #
    MaxNumNodes_filename = LastStep_OutputFolderName + "MaxNumNodes.txt"
    max_nodes = np.loadtxt(MaxNumNodes_filename, dtype = 'int64', delimiter = "\t").item()
    
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
    
    dataset = SpatialOmicsImageDataset(LastStep_OutputFolderName, transform=T.ToDense(max_nodes))
    
    
    class Net(torch.nn.Module):
        def __init__(self, in_channels, out_channels, hidden_channels=Embedding_Dimension):
            super(Net, self).__init__()
    
            self.conv1 = DenseGraphConv(in_channels, hidden_channels)
            num_cluster1 = Num_TCN   #This is a hyperparameter.
            self.pool1 = Linear(hidden_channels, num_cluster1)
    
        def forward(self, x, adj, mask=None):
    
            x = F.relu(self.conv1(x, adj, mask))
            s = self.pool1(x)  #Here "s" is a non-softmax tensor.
            x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)
            #Save important clustering results_1.
            ClusterAssignTensor_1 = s
            ClusterAdjTensor_1 = adj
    
            return F.log_softmax(x, dim=-1), mc1, o1, ClusterAssignTensor_1, ClusterAdjTensor_1
    
    
    def train(epoch):
        model.train()
        loss_all = 0
    
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, mc_loss, o_loss, _, _ = model(data.x, data.adj, data.mask)
            loss = mc_loss + o_loss
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
        return loss_all
    
    
    # Extract a single graph for TCN learning.
    ThisStep_OutputFolderName = "/home/owkin/project/cytocommunity_results/fractions/experiments/"+ Image_Name + "/"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ThisStep_OutputFolderName = os.path.join(ThisStep_OutputFolderName, timestamp, "crossval")
    os.makedirs(ThisStep_OutputFolderName, exist_ok=True)
    
    # Save a copy of the config file to the output directory
    output_config_path = os.path.join(ThisStep_OutputFolderName, "config.yaml")
    with open(output_config_path, "w") as f:
        yaml.dump(config, f)
    
    train_index = [region_name_list["Image"].values.tolist().index(Image_Name)]
    train_dataset = dataset[train_index]
    train_loader = DenseDataLoader(train_dataset, batch_size=1, pin_memory=True, num_workers=0)
    all_sample_loader = DenseDataLoader(train_dataset, batch_size=1, pin_memory=True, num_workers=0)
    
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    run_number = 1
    while run_number <= Num_Run:  #Generate multiple independent runs for ensemble.
    
        print(f"This is Run{run_number:02d}")
    
        model = Net(dataset.num_features, 1).to(device)  #Initializing the model.
        optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)
        
        RunFolderName = ThisStep_OutputFolderName + "/Run" + str(run_number)
        if os.path.exists(RunFolderName):
            shutil.rmtree(RunFolderName)
        os.makedirs(RunFolderName)  #Creating the Run folder.
        
        filename_0 = RunFolderName + "/Epoch_UnsupervisedLoss.csv"
        headers_0 = ["Epoch", "UnsupervisedLoss"]
        with open(filename_0, "w", newline='') as f0:
            f0_csv = csv.writer(f0)
            f0_csv.writerow(headers_0)
    
        previous_loss = float("inf")  #Initialization.
        for epoch in tqdm(range(1, Num_Epoch+1)):  #Specify the number of epoch in each independent run.
            train_loss = train(epoch)
    
            #print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}")
            with open(filename_0, "a", newline='') as f0:
                f0_csv = csv.writer(f0)
                f0_csv.writerow([epoch, train_loss])
            
            if train_loss == 0 and train_loss == previous_loss:  #If two consecutive losses are both zeros, the learning gets stuck.
                break  #stop the training.
            else:
                previous_loss = train_loss
    
        print(f"Final train loss is {train_loss:.4f}")
        if train_loss >= Loss_Cutoff:   #This is an empirical cutoff of the final loss to avoid underfitting.
            shutil.rmtree(RunFolderName)  #Remove the specific folder and all files inside it for re-creating the Run folder.
            continue  #restart this run.
    
        #Extract the soft TCN assignment matrix using the trained model.
        for EachData in all_sample_loader:
            EachData = EachData.to(device)
            TestModelResult = model(EachData.x, EachData.adj, EachData.mask)
    
            ClusterAssignMatrix1 = TestModelResult[3][0, :, :]
            ClusterAssignMatrix1 = torch.softmax(ClusterAssignMatrix1, dim=-1)  #Checked, consistent with the built-in function "dense_mincut_pool".
            ClusterAssignMatrix1 = ClusterAssignMatrix1.cpu().detach().numpy()
            filename1 = RunFolderName + "/TCN_AssignMatrix1.csv"
            np.savetxt(filename1, ClusterAssignMatrix1, delimiter=',')
    
            ClusterAdjMatrix1 = TestModelResult[4][0, :, :]
            ClusterAdjMatrix1 = ClusterAdjMatrix1.cpu().detach().numpy()
            filename2 = RunFolderName + "/TCN_AdjMatrix1.csv"
            np.savetxt(filename2, ClusterAdjMatrix1, delimiter=',')
    
            NodeMask = EachData.mask
            NodeMask = np.array(NodeMask.cpu())
            filename3 = RunFolderName + "/NodeMask.csv"
            np.savetxt(filename3, NodeMask.T, delimiter=',', fmt='%i')  #save as integers.
    
        run_number = run_number + 1

    return timestamp

