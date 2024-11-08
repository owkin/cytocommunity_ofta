from sklearn.model_selection import StratifiedKFold
import torch
from torch_geometric.loader import DenseDataLoader
import torch_geometric.transforms as T
import os
import shutil
import numpy as np
import datetime
import csv
import yaml
from tqdm import tqdm
from cytocommunity_ofta.utils.data_utils import SpatialOmicsImageDataset
from cytocommunity_ofta.utils.model import Net
from cytocommunity_ofta.utils.train_test import train, test
from cytocommunity_ofta.utils.constants import device
import random


def set_seed(seed):
    random.seed(seed)  # Set random seed for Python's random module
    np.random.seed(seed)  # Set random seed for NumPy
    torch.manual_seed(seed)  # Set random seed for PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs (if using CUDA)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning

# Set seed for reproducibility
seed_value = 42  # You can choose any integer
set_seed(seed_value)

# Load hyperparameters from config file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Access hyperparameters and paths from the config file
hyperparams = config['hyperparameters']
paths = config['paths']

Num_TCN = hyperparams['Num_TCN']
Num_Times = hyperparams['Num_Times']
Num_Folds = hyperparams['Num_Folds']
Num_Epoch = hyperparams['Num_Epoch']
Embedding_Dimension = hyperparams['Embedding_Dimension']
LearningRate = hyperparams['LearningRate']
MiniBatchSize = hyperparams['MiniBatchSize']
WeightDecay = hyperparams['WeightDecay']
Dropout = hyperparams['Dropout']
beta = hyperparams['beta']

LastStep_OutputFolderName = paths['LastStep_OutputFolderName']
ThisStep_OutputFolderName = paths['ThisStep_OutputFolderName']

# Load dataset
MaxNumNodes_filename = LastStep_OutputFolderName + "MaxNumNodes.txt"
max_nodes = np.loadtxt(MaxNumNodes_filename, dtype="int64", delimiter="\t").item()
dataset = SpatialOmicsImageDataset(
    LastStep_OutputFolderName, transform=T.ToDense(max_nodes)
)

# Extract labels for stratification
labels = np.array([data.y.item() for data in dataset])

# Date and time for folder naming
dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ThisStep_OutputFolderName = os.path.join(ThisStep_OutputFolderName, dt, "crossval")
os.makedirs(ThisStep_OutputFolderName, exist_ok=True)

# Save a copy of the config file to the output directory
output_config_path = os.path.join(ThisStep_OutputFolderName, "config.yaml")
with open(output_config_path, "w") as f:
    yaml.dump(config, f)

for num_time in range(1, Num_Times + 1):  # Repeat cross-validation multiple times
    print(f"This is time: {num_time:02d}")
    TimeFolderName = os.path.join(ThisStep_OutputFolderName, f"Time{num_time}")
    if os.path.exists(TimeFolderName):
        shutil.rmtree(TimeFolderName)
    os.makedirs(TimeFolderName)

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=Num_Folds, shuffle=True, random_state=num_time)
    
    for num_fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels), start=1):
        print(f"This is fold: {num_fold:02d}")

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DenseDataLoader(train_dataset, batch_size=MiniBatchSize, shuffle=True, pin_memory=True, num_workers=32)
        test_loader = DenseDataLoader(test_dataset, batch_size=1, pin_memory=True, num_workers=32)

        model = Net(dataset.num_features, dataset.num_classes, Embedding_Dimension, Num_TCN, Dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate, weight_decay=WeightDecay)  # Adjust weight_decay as needed


        FoldFolderName = os.path.join(TimeFolderName, f"Fold{num_fold}")
        os.makedirs(FoldFolderName)
        filename_0 = os.path.join(FoldFolderName, "Epoch_TrainLoss.csv")
        headers_0 = ["Epoch", "TrainLoss", "TestAccuracy", "TrainLoss_CE", "TrainLoss_MinCut", "TrainingAUC", "TestAUC", "TestLoss_MinCut"]

        with open(filename_0, "w", newline="") as f0:
            f0_csv = csv.writer(f0)
            f0_csv.writerow(headers_0)

        for epoch in tqdm(range(1, Num_Epoch + 1)):
            train_loss, train_loss_CE, train_loss_MinCut, train_auc = train(model, train_loader, optimizer, beta, train_dataset)
            test_acc, test_pr, test_auc, test_loss_MinCut = test(test_loader, model)

            with open(filename_0, "a", newline="") as f0:
                f0_csv = csv.writer(f0)
                f0_csv.writerow([epoch, train_loss, test_acc, train_loss_CE, train_loss_MinCut, train_auc, test_auc, test_loss_MinCut])

        print(f"Final train loss is {train_loss:.4f} with loss_CE of {train_loss_CE:.4f} and loss_MinCut of {train_loss_MinCut:.4f}, and final test accuracy is {test_acc:.4f} and train auc {train_auc:.4f} and test auc {test_auc:.4f}")

        filename6 = FoldFolderName + "/TestSet_Pr_Pred_Truth.csv"
        np.savetxt(filename6, test_pr, delimiter=',')

        #Extract the soft clustering matrix using the trained model of each fold.
        all_sample_loader = DenseDataLoader(dataset, batch_size=1, pin_memory=True, num_workers=32)
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
            TrueLableArray = np.array(EachData.y.view(-1))
            PredLabelArray = np.array(PredLabel)
            #print(f'Prediction correct flag: {CorrectFlag:01d}, True label: {TrueLableArray}, Predicted label: {PredLabelArray}')
            with open(filename_5, "a", newline='') as f5:
                f5_csv = csv.writer(f5)
                f5_csv.writerow([EachSample_num, CorrectFlag, TrueLableArray, PredLabelArray])

            ClusterAssignMatrix1 = TestModelResult[3][0, :, :]
            ClusterAssignMatrix1 = torch.softmax(ClusterAssignMatrix1, dim=-1)  #checked, consistent with the function built in "dense_mincut_pool".
            ClusterAssignMatrix1 = ClusterAssignMatrix1.detach().numpy()
            filename1 = FoldFolderName + "/ClusterAssignMatrix1_" + str(EachSample_num) + ".csv"
            np.savetxt(filename1, ClusterAssignMatrix1, delimiter=',')

            ClusterAdjMatrix1 = TestModelResult[4][0, :, :]
            ClusterAdjMatrix1 = ClusterAdjMatrix1.detach().numpy()
            filename2 = FoldFolderName + "/ClusterAdjMatrix1_" + str(EachSample_num) + ".csv"
            np.savetxt(filename2, ClusterAdjMatrix1, delimiter=',')

            NodeMask = EachData.mask
            NodeMask = np.array(NodeMask)
            filename3 = FoldFolderName + "/NodeMask_" + str(EachSample_num) + ".csv"
            np.savetxt(filename3, NodeMask.T, delimiter=',', fmt='%i')  #save as integers.

            EachSample_num = EachSample_num + 1
