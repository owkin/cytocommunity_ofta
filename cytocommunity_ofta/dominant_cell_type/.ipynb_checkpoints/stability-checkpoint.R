library(diceR)


args <- commandArgs(trailingOnly = TRUE)


## Hyperparameters
Image_Name <- args[1]
timestamp <- args[2]


#Import data.
LastStep_OutputFolderName <- paste0("/home/owkin/project/cytocommunity_results/dominant_cell_type/experiments/", Image_Name, "/", timestamp, "/crossval/")
NodeMask <- read.csv(paste0(LastStep_OutputFolderName, "Run1/NodeMask.csv"), header = FALSE)
nonzero_ind <- which(NodeMask$V1 == 1)

#Find the file names of all soft TCN assignment matrices.
allSoftClustFile <- list.files(path = LastStep_OutputFolderName, pattern = "TCN_AssignMatrix1.csv", recursive = TRUE)
allHardClustLabel <- vector()

for (i in 1:length(allSoftClustFile)) {
  print(i)
  for (j in 1:i):{
  
  ClustMatrix <- read.csv(paste0(LastStep_OutputFolderName, allSoftClustFile[i]), header = FALSE, sep = ",")
  ClustMatrix <- ClustMatrix[nonzero_ind,]
  HardClustLabel <- apply(as.matrix(ClustMatrix), 1, which.max)
  rm(ClustMatrix)
  
  allHardClustLabel <- cbind(allHardClustLabel, as.vector(HardClustLabel))
  
} #end of for.

finalClass <- diceR::majority_voting(allHardClustLabel, is.relabelled = FALSE)

ThisStep_OutputFolderName <- paste0("/home/owkin/project/cytocommunity_results/dominant_cell_type/experiments/", Image_Name, "/", timestamp, "/ensemble/")
if (file.exists(ThisStep_OutputFolderName)){
    unlink(ThisStep_OutputFolderName, recursive=TRUE)  #delete the folder if already exists.
} 
dir.create(ThisStep_OutputFolderName)

write.table(finalClass, file = paste0(ThisStep_OutputFolderName, "/TCNLabel_MajorityVoting_nruns_", i, ".csv"), append = FALSE, quote = FALSE, row.names = FALSE, col.names = FALSE)
}

