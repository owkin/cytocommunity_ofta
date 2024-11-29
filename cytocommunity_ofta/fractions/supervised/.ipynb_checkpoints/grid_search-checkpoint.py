import yaml
from cytocommunity_ofta.fractions.supervised.tcn_learning_supervised import tcn_learning_supervised
from cytocommunity_ofta.fractions.supervised.result_visualization import result_visualization
import subprocess
import yaml
from loguru import logger


for num_tcns in [3,4,6,7,8,9]:
    data = {
        "graph": {
            "neighbors_order": 1
        },
        "hyperparameters": {
            "Num_TCN": num_tcns,
            "Num_Times": 3,
            "Num_Epoch": 50,
            "Num_Folds": 5,
            "Embedding_Dimension": 128,
            "LearningRate": 0.01,
            "MiniBatchSize": 16,
            "beta": 1
        }
    }
    
    # Write to a YAML file
    with open("config.yaml", "w") as file:
        yaml.dump(data, file, default_flow_style=False)
    
    logger.info("tcn_learning_supervised")
    timestamp = tcn_learning_supervised()
    logger.info("tcn_ensemble")
    
    command = [
        "Rscript",           # Call Rscript to run an R script
        "/home/owkin/cytocommunity_ofta/cytocommunity_ofta/fractions/supervised/tcn_ensemble.R",    # Path to your R script
        timestamp          
    ]
    
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    
    logger.info("result_visualization")
    result_visualization(timestamp)
