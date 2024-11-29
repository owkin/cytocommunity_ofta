from cytocommunity_ofta.fractions.supervised.tcn_learning_supervised import tcn_learning_supervised
from cytocommunity_ofta.fractions.supervised.result_visualization import result_visualization
import subprocess
import yaml
from loguru import logger


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