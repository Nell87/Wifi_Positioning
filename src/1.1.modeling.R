# Load previous code
setwd("C:/SARA/PORTFOLIO/Wifi_Positioning/src/")
source("0.1.initial_exploration_preprocess.R") # Change it to read new prepared dataset

#### A. PREPARING FOR MODELING #### -----------------------------------------------------------------------------

# Split Data before modeling 
Data_FullSplit<-split(data_full, data_full$source)
list2env(Data_FullSplit, envir=.GlobalEnv)
rm(Data_FullSplit)

# Create a DataValidOriginal for analyzing performance 
df_datavalid_orig<-df_datavalid

# Prepare Parallel Process
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

# 10 fold cross validation    
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, allowParallel = TRUE)





