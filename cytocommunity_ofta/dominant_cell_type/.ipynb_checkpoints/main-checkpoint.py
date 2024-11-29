from cytocommunity_ofta.dominant_cell_type.tcnlearning_unsupervised import tcn_learning_unsupervised
from cytocommunity_ofta.dominant_cell_type.result_visualization import result_visualization
import subprocess
import yaml
from loguru import logger


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
    
## Hyperparameters
graph = config["graph"]
hyperparams = config["hyperparameters"]
InputFolderName = "/home/owkin/project/cytocommunity_results/dominant_cell_type/raw/"
Image_Name_All = graph["Image_Name"]
Num_TCN = hyperparams["Num_TCN"]
Num_Run = hyperparams["Num_Run"]
Num_Epoch = hyperparams["Num_Epoch"]
Embedding_Dimension = hyperparams["Embedding_Dimension"]
Learning_Rate = hyperparams["LearningRate"]

for Image_Name in Image_Name_All:

    logger.info("tcn_learning_unsupervised")
    
    timestamp = tcn_learning_unsupervised(Image_Name)
    
    logger.info("tcn_ensemble")
    
    command = [
        "Rscript",           # Call Rscript to run an R script
        "tcn_ensemble.R",    # Path to your R script
        Image_Name,          # First argument: input file
        timestamp          # Second argument: output file
    ]
    
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    
    logger.info("result_visualization")
    
    result_visualization(Image_Name, timestamp)