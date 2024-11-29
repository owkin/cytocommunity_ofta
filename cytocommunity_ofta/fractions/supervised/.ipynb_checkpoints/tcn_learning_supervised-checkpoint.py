import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGraphConv, dense_mincut_pool
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
import os
import shutil
import numpy as np
import datetime
import csv
import random
import yaml
from tqdm import tqdm


def tcn_learning_supervised():
        
    device = torch.device("cuda:1")
        
    # Load hyperparameters from config file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("config", config)
    
    Num_TCN = config["hyperparameters"]["Num_TCN"]
    Num_Times = config["hyperparameters"]["Num_Times"]
    Num_Epoch = config["hyperparameters"]["Num_Epoch"]
    Num_Folds = config["hyperparameters"]["Num_Folds"]
    Embedding_Dimension = config["hyperparameters"]["Embedding_Dimension"]
    LearningRate = config["hyperparameters"]["LearningRate"]
    MiniBatchSize = config["hyperparameters"]["MiniBatchSize"]
    beta = config["hyperparameters"]["beta"]
    
    
    ## Load dataset from the constructed Dataset.
    #LastStep_OutputFolderName = "./Step1_Output/"
    LastStep_OutputFolderName = "/home/owkin/project/cytocommunity_results/fractions/supervised/graphs/neighbors_order_1/"
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
    
            self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)
    
            self.lin1 = Linear(hidden_channels, hidden_channels)
            self.lin2 = Linear(hidden_channels, out_channels)
    
        def forward(self, x, adj, mask=None):
    
            x = F.relu(self.conv1(x, adj, mask))
            s = self.pool1(x)  #here s is a non-softmax tensor.
            x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)
            #Save important clustering results_1.
            ClusterAssignTensor_1 = s
            ClusterAdjTensor_1 = adj
    
            x = self.conv3(x, adj)
    
            x = x.mean(dim=1)
            x = F.relu(self.lin1(x))
            x = self.lin2(x)
            return F.log_softmax(x, dim=-1), mc1, o1, ClusterAssignTensor_1, ClusterAdjTensor_1
    
    
    def train(epoch):
        model.train()
        loss_all = 0
        loss_CE_all = 0
        loss_MinCut_all = 0
    
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, mc_loss, o_loss, _, _ = model(data.x, data.adj, data.mask)
            loss_CE = F.nll_loss(out, data.y.view(-1))
            loss_MinCut = mc_loss + o_loss
    
            #loss = F.nll_loss(out, data.y.view(-1)) * (1 - beta) + (mc_loss + o_loss) * beta
            loss = loss_CE * (1 - beta) + loss_MinCut * beta
            loss.backward()
            loss_all += data.y.size(0) * loss.item()  #total running loss for a mini-batch.
            loss_CE_all += data.y.size(0) * loss_CE.item()
            loss_MinCut_all += data.y.size(0) * loss_MinCut.item()
            optimizer.step()
        return loss_all / len(train_dataset), loss_CE_all / len(train_dataset), loss_MinCut_all / len(train_dataset)  #average sample loss for this particular epoch.
    
    
    @torch.no_grad()
    def test(loader):
        model.eval()
        correct = 0
        pr_Table = np.zeros([1,4]) #initializing an array.
        loss_MinCut_all = 0
    
        for data in loader:
            data = data.to(device)
            #pred = model(data.x, data.adj, data.mask)[0].max(dim=1)[1]
            ModelResultPr, mc_loss, o_loss, _, _ = model(data.x, data.adj, data.mask)
            pred = ModelResultPr.max(dim=1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()

            # MinCut loss
            loss_MinCut = mc_loss + o_loss
            loss_MinCut_all += data.y.size(0) * loss_MinCut.item()
            
            pred_info = np.column_stack(
                (
                    np.array(torch.exp(ModelResultPr).cpu()),
                    np.array(pred.cpu()),
                    np.array(data.y.view(-1).cpu()),
                )
            )
            pr_Table = np.row_stack((pr_Table, pred_info))
    
        return correct / len(loader.dataset), pr_Table, loss_MinCut_all / len(loader.dataset)
    
    
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ThisStep_OutputFolderName = "/home/owkin/project/cytocommunity_results/fractions/supervised/experiments/"
    ThisStep_OutputFolderName = os.path.join(ThisStep_OutputFolderName, timestamp, "crossval")
    os.makedirs(ThisStep_OutputFolderName, exist_ok=True)
    
    for num_time in range(1, Num_Times+1):  #10 times of k-fold cross-validation.
        print(f'This is time: {num_time:02d}')
        TimeFolderName = ThisStep_OutputFolderName + "/Time" + str(num_time)
        if os.path.exists(TimeFolderName):
            shutil.rmtree(TimeFolderName)
        os.makedirs(TimeFolderName)  #Creating the Time folder.
    
        ###Below is for 10-fold cross-validation to evaluate model performance.
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
        n_partition = len(dataset) // Num_Folds   #number of test samples in the 10-fold cross-validation.
        test_pool = set(np.arange(0, len(dataset)))   #test_pool initialization for each Time.
        for num_fold in range(1, Num_Folds+1):  #sample split for the 10-fold cross-validation.
            if num_fold == Num_Folds:
                n_test = len(dataset) - (n_partition*(Num_Folds-1))
            else:
                n_test = n_partition
            
            test_list = random.sample(test_pool, n_test)
            test_pool = test_pool.difference(set(test_list))  #update test pool by removing used test samples.
    
            train_list = list(set(np.arange(0, len(dataset))).difference(set(test_list)))
            
            print(f'This is fold: {num_fold:02d}, TestSamples: {test_list}')
            test_dataset = dataset[test_list]
            train_dataset = dataset[train_list]
            train_loader = DenseDataLoader(train_dataset, batch_size=MiniBatchSize, shuffle=True)
            test_loader = DenseDataLoader(test_dataset, batch_size=1)
    
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = Net(dataset.num_features, dataset.num_classes).to(device)  #Initialize model for each fold.
            optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate)
    
            FoldFolderName = TimeFolderName + "/Fold" + str(num_fold)
            os.makedirs(FoldFolderName)  #Creating the Fold folder.
            filename_0 = FoldFolderName + "/Epoch_TrainLoss.csv"
            headers_0 = ["Epoch", "TrainLoss", "TestAccuracy", "TrainLoss_CE", "TrainLoss_MinCut", "TestLoss_MinCut"]
            with open(filename_0, "w", newline='') as f0:
                f0_csv = csv.writer(f0)
                f0_csv.writerow(headers_0)
    
            for epoch in tqdm(range(1, Num_Epoch+1)):     #Specify the number of epoch for training in each fold.
                train_loss, train_loss_CE, train_loss_MinCut = train(epoch)
                test_acc, test_pr, test_loss_MinCut = test(test_loader)
    
                #print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}')
                with open(filename_0, "a", newline='') as f0:
                    f0_csv = csv.writer(f0)
                    f0_csv.writerow([epoch, train_loss, test_acc, train_loss_CE, train_loss_MinCut, test_loss_MinCut])
    
            print(f"Final train loss is {train_loss:.4f} with loss_CE of {train_loss_CE:.4f} and loss_MinCut of {train_loss_MinCut:.4f}, and final test accuracy is {test_acc:.4f}")
            #print(test_pr)
            filename6 = FoldFolderName + "/TestSet_Pr_Pred_Truth.csv"
            np.savetxt(filename6, test_pr, delimiter=',')
    
            #Extract the soft clustering matrix using the trained model of each fold.
            all_sample_loader = DenseDataLoader(dataset, batch_size=1)
            EachSample_num = 0
            
            filename_5 = FoldFolderName + "/ModelPrediction.csv"
            headers_5 = ["SampleNum", "PredictionCorrectFlag", "TrueLabel", "PredictedLabel"]
            with open(filename_5, "w", newline='') as f5:
                f5_csv = csv.writer(f5)
                f5_csv.writerow(headers_5)
    
            for EachData in all_sample_loader:
                EachData = EachData.to(device)
                TestModelResult = model(EachData.x, EachData.adj, EachData.mask)
                PredLabel = TestModelResult[0].max(dim=1)[1]
                CorrectFlag = PredLabel.eq(EachData.y.view(-1)).sum().item()
                TrueLableArray = np.array(EachData.y.view(-1).cpu())
                PredLabelArray = np.array(PredLabel.cpu())
                #print(f'Prediction correct flag: {CorrectFlag:01d}, True label: {TrueLableArray}, Predicted label: {PredLabelArray}')
                with open(filename_5, "a", newline='') as f5:
                    f5_csv = csv.writer(f5)
                    f5_csv.writerow([EachSample_num, CorrectFlag, TrueLableArray, PredLabelArray])
    
                ClusterAssignMatrix1 = TestModelResult[3][0, :, :]
                ClusterAssignMatrix1 = torch.softmax(ClusterAssignMatrix1, dim=-1)  #checked, consistent with the function built in "dense_mincut_pool".
                ClusterAssignMatrix1 = ClusterAssignMatrix1.cpu().detach().numpy()
                filename1 = FoldFolderName + "/ClusterAssignMatrix1_" + str(EachSample_num) + ".csv"
                np.savetxt(filename1, ClusterAssignMatrix1, delimiter=',')
    
                ClusterAdjMatrix1 = TestModelResult[4][0, :, :]
                ClusterAdjMatrix1 = ClusterAdjMatrix1.cpu().detach().numpy()
                filename2 = FoldFolderName + "/ClusterAdjMatrix1_" + str(EachSample_num) + ".csv"
                np.savetxt(filename2, ClusterAdjMatrix1, delimiter=',')
    
                NodeMask = EachData.mask
                NodeMask = np.array(NodeMask.cpu())
                filename3 = FoldFolderName + "/NodeMask_" + str(EachSample_num) + ".csv"
                np.savetxt(filename3, NodeMask.T, delimiter=',', fmt='%i')  #save as integers.
    
                EachSample_num = EachSample_num + 1
    
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return timestamp

