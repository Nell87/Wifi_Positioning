#### 0. INCLUDES  --------------------------------------------------------------------
#source("0.1.initial_exploration_preprocess.R") 

#Load Libraries: p_load can install, load,  and update packages
if(require("pacman")=="FALSE"){
  install.packages("pacman")
} 

pacman::p_load(dplyr, lubridate, caret)

# Setwd
setwd("C:/SARA/PORTFOLIO/Wifi_Positioning/data/")

# Load Data
df_datatrain <- read.csv2("trainingData_prepared.csv", header=TRUE, sep=",",  stringsAsFactors=FALSE, na.strings=c("NA", "-", "?"))
df_datavalid<-read.csv2("validationData_prepared.csv", header=TRUE, sep=",",  stringsAsFactors=FALSE, na.strings=c("NA", "-", "?"))

#### A. PREPARING FOR MODELING #### -----------------------------------------------------------------------------
# Create a DataValidOriginal for analyzing performance 
df_datavalid_orig<-df_datavalid

# Prepare Parallel Process
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

# 10 fold cross validation    
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, allowParallel = TRUE)





