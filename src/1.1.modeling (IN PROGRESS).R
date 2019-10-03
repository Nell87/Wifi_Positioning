#### 0. INCLUDES  --------------------------------------------------------------------
#source("0.1.initial_exploration_preprocess.R") 

#Load Libraries: p_load can install, load,  and update packages
if(require("pacman")=="FALSE"){
  install.packages("pacman")
} 

pacman::p_load(rstudioapi,dplyr, lubridate, caret,parallel,doParallel,
               randomForest, class,e1071, reshape2,RColorBrewer)

# Setwd (1º current wd where is the script, then we move back to the 
# general folder)
current_path = getActiveDocumentContext()$path 
setwd(dirname(current_path))
setwd("..")
rm(current_path)

# Load Data
df_datatrain <- read.csv("data/trainingData_prepared.csv", 
                          stringsAsFactors=FALSE, row.names = NULL,
                          na.strings=c("NA", "-", "?"))

df_datavalid<-read.csv("data/validationData_prepared.csv", 
                         stringsAsFactors=FALSE, row.names = NULL,
                         na.strings=c("NA", "-", "?"), )

# Transform some variables to factor/numeric/datetime
factors<-c("FLOOR", "BUILDINGID", "RELATIVEPOSITION", "USERID", "PHONEID")
df_datatrain[,factors]<-lapply(df_datatrain[,factors], as.factor)
df_datavalid[,factors]<-lapply(df_datavalid[,factors], as.factor)
rm(factors)

numeric<-c("LONGITUDE", "LATITUDE")
df_datatrain[,numeric]<-lapply(df_datatrain[,numeric], as.numeric)
df_datavalid[,numeric]<-lapply(df_datavalid[,numeric], as.numeric)

rm(numeric)

#### A. PREPARING FOR MODELING #### -----------------------------------------------------------------------------
# Change some names for analyzing performance 
df_datavalid <- df_datavalid %>% rename(BUILDINGID_orig = BUILDINGID, 
                                        LONGITUDE_orig = LONGITUDE,
                                        LATITUDE_orig=LATITUDE, FLOOR_orig=FLOOR)


# Prepare Parallel Process
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

# 10 fold cross validation    
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, 
                           allowParallel = TRUE)

WAPS<-grep("WAP", names(df_datatrain), value=T)

#### B. PREDICTING BUILDING ####----------------------------------------------------------------------
# Train dt model        <-- 38.67 sec elapsed       Accuracy:1  Kappa: 1
# system.time(Building_SVM01<-caret::train(BUILDINGID~HighWAP, data= df_datatrain, method="svmLinear", 
#                                trControl=fitControl))
# saveRDS(Building_SVM01, file = "./models/Building_SVM01.rds")

Building_SVM01<-readRDS("./models/Building_SVM01.rds")
PredictorsBuild<-predict(Building_SVM01, df_datavalid)
ConfusionMatrix<-confusionMatrix(PredictorsBuild, df_datavalid$BUILDINGID_orig) # Accuracy:1 Kappa:1

ConfusionMatrix

# Add building Predictions in DataValidation
df_datavalid$BUILDINGID<-PredictorsBuild

rm(ConfusionMatrix, Building_SVM01, PredictorsBuild)

#### C. PREDICTING FLOOR PER BUILDING (1 model per building) ####-------------------------------------------------------------
WAPS<-grep("^[^HighWAP]*[WAP][^HighWAP]*$", names(df_datatrain), value=T)

WAPS<-setdiff(grep('WAP', names(df_datatrain),value=T), 
              grep('HighWAP', names(df_datatrain),value=T))

# Split DataTrain & Valid per Building
Buildings<-split(df_datatrain, df_datatrain$BUILDINGID)
names(Buildings)<-c("dt_b0", "dt_b1", "dt_b2")
list2env(Buildings, envir = .GlobalEnv)

Buildings<-split(df_datavalid, df_datavalid$BUILDINGID)
names(Buildings)<-c("dv_b0", "dv_b1", "dv_b2")
list2env(Buildings, envir = .GlobalEnv)
rm(Buildings)

##### C.1. Floor Building 0 ##### 
# Reset the levels
dt_b0$FLOOR<-as.factor(as.character(dt_b0$FLOOR))
dv_b0$FLOOR_orig<-as.factor(as.character(dv_b0$FLOOR_orig))
levels(dt_b0$FLOOR) # 4 levels

# Random forest ________________________________________________________________
# bestmtry_dt_b0<-tuneRF(dt_b0[WAPS], dt_b0$FLOOR, ntreeTry=100, stepFactor=2, 
#                      improve=0.05,trace=TRUE, plot=T)   # <- 34

# system.time(B0_floor_rf<-randomForest(y=dt_b0$FLOOR, x=dt_b0[WAPS], 
#                                       importance=T,maximize=T,
#                                        method="rf", trControl=fitControl,
#                                        ntree=100, mtry=34,allowParalel=TRUE))

# saveRDS(B0_floor_rf, file = "./models/B0_floor_rf.rds")

B0_floor_rf<-readRDS("./models/B0_floor_rf.rds")
Predictors_B0_floor<-predict(B0_floor_rf, dv_b0)
ConfusionMatrix<-confusionMatrix(Predictors_B0_floor, dv_b0$FLOOR_orig) 
ConfusionMatrix

rm(B0_floor_rf,Predictors_B0_floor, ConfusionMatrix)

# KNN __________________________________________________________________________    
system.time(floor_b0_knn_pred<- knn(train = dt_b0[1:311], 
                               test = dv_b0[1:311], 
                               cl = dt_b0$FLOOR))

ConfusionMatrix<-confusionMatrix(floor_b0_knn_pred, dv_b0$FLOOR_orig) 
ConfusionMatrix
rm(ConfusionMatrix, floor_b0_knn_pred)

# SVM __________________________________________________________________________
# system.time(B0_floor_svm <- svm(y = dt_b0$FLOOR, x=dt_b0[WAPS], kernel = "linear"))
# saveRDS(B0_floor_svm, file = "./models/B0_floor_svm.rds")

B0_floor_svm<-readRDS("./models/B0_floor_svm.rds")
Predictors_B0_floor<-predict(B0_floor_svm, dv_b0[WAPS])
ConfusionMatrix<-confusionMatrix(Predictors_B0_floor, dv_b0$FLOOR_orig) 
ConfusionMatrix

rm(B0_floor_svm,Predictors_B0_floor, ConfusionMatrix)

##### C.2. Floor Building 1 ##### 
# Reset the levels
dt_b1$FLOOR<-as.factor(as.character(dt_b1$FLOOR))
dv_b1$FLOOR_orig<-as.factor(as.character(dv_b1$FLOOR_orig))
levels(dt_b1$FLOOR) # 4 levels

# Random forest ________________________________________________________________
# bestmtry_dt_b1<-tuneRF(dt_b1[WAPS], dt_b1$FLOOR, ntreeTry=100, stepFactor=2, 
#                        improve=0.05,trace=TRUE, plot=T)   # <- 68

# system.time(B1_floor_rf<-randomForest(y=dt_b1$FLOOR, x=dt_b1[WAPS], 
#                                        importance=T,maximize=T,
#                                         method="rf", trControl=fitControl,
#                                         ntree=100, mtry=68,allowParalel=TRUE))

# saveRDS(B1_floor_rf, file = "./models/B1_floor_rf.rds")

B1_floor_rf<-readRDS("./models/b1_floor_rf.rds")
Predictors_b1_floor<-predict(B1_floor_rf, dv_b1)
ConfusionMatrix<-confusionMatrix(Predictors_b1_floor, dv_b1$FLOOR_orig) 
ConfusionMatrix

rm(B1_floor_rf,Predictors_b1_floor, ConfusionMatrix)

# KNN __________________________________________________________________________    
system.time(floor_b1_knn_pred<- knn(train = dt_b1[1:311], 
                                    test = dv_b1[1:311], 
                                    cl = dt_b1$FLOOR))

ConfusionMatrix<-confusionMatrix(floor_b1_knn_pred, dv_b1$FLOOR_orig) 
ConfusionMatrix
rm(ConfusionMatrix, floor_b1_knn_pred)

# SVM __________________________________________________________________________
# system.time(B1_floor_svm <- svm(y = dt_b1$FLOOR, x=dt_b1[WAPS], kernel = "linear"))
# saveRDS(B1_floor_svm, file = "./models/B1_floor_svm.rds")

B1_floor_svm<-readRDS("./models/B1_floor_svm.rds")
Predictors_b1_floor<-predict(B1_floor_svm, dv_b1[WAPS])
ConfusionMatrix<-confusionMatrix(Predictors_b1_floor, dv_b1$FLOOR_orig) 
ConfusionMatrix

rm(B1_floor_svm,Predictors_b1_floor, ConfusionMatrix)

##### C.3. Floor Building 2 ##### 
# Reset the levels
dt_b2$FLOOR<-as.factor(as.character(dt_b2$FLOOR))
dv_b2$FLOOR_orig<-as.factor(as.character(dv_b2$FLOOR_orig))
levels(dt_b2$FLOOR) # 4 levels

# Random forest ________________________________________________________________
# bestmtry_dt_b2<-tuneRF(dt_b2[WAPS], dt_b2$FLOOR, ntreeTry=100, stepFactor=2, 
#                         improve=0.05,trace=TRUE, plot=T)   # <- 34

# system.time(B2_floor_rf<-randomForest(y=dt_b2$FLOOR, x=dt_b2[WAPS], 
#                                         importance=T,maximize=T,
#                                          method="rf", trControl=fitControl,
#                                          ntree=100, mtry=34,allowParalel=TRUE))

# saveRDS(B2_floor_rf, file = "./models/B2_floor_rf.rds")

B2_floor_rf<-readRDS("./models/B2_floor_rf.rds")
Predictors_b2_floor<-predict(B2_floor_rf, dv_b2)
ConfusionMatrix<-confusionMatrix(Predictors_b2_floor, dv_b2$FLOOR_orig) 
ConfusionMatrix

rm(B2_floor_rf,Predictors_b2_floor, ConfusionMatrix)

# KNN __________________________________________________________________________    
system.time(floor_b2_knn_pred<- knn(train = dt_b2[1:311], 
                                    test = dv_b2[1:311], 
                                    cl = dt_b2$FLOOR))

ConfusionMatrix<-confusionMatrix(floor_b2_knn_pred, dv_b2$FLOOR_orig) 
ConfusionMatrix
rm(ConfusionMatrix, floor_b2_knn_pred)

# SVM __________________________________________________________________________
system.time(B2_floor_svm <- svm(y = dt_b2$FLOOR, x=dt_b2[WAPS], kernel = "linear"))
saveRDS(B2_floor_svm, file = "./models/B2_floor_svm.rds")

B2_floor_svm<-readRDS("./models/B2_floor_svm.rds")
Predictors_b2_floor<-predict(B2_floor_svm, dv_b2[WAPS])
ConfusionMatrix<-confusionMatrix(Predictors_b2_floor, dv_b2$FLOOR_orig) 
ConfusionMatrix

rm(B2_floor_svm,Predictors_b2_floor, ConfusionMatrix)

#### D.PREDICTING LONGITUDE PER BUILDING (1 model per building) #### 
WAPS<- intersect(grep("WAP",names(dt_b0),value=T),
          grep("High",names(dt_b0),invert=TRUE,value=T))

##### D.1. Longitude building 0 #####
# Random forest ________________________________________________________________
# bestmtry_dt_b0<-tuneRF(dt_b0[WAPS], dt_b0$LONGITUDE, ntreeTry=100, stepFactor=2, 
#                       improve=0.05,trace=TRUE, plot=T)   # <- 52

# system.time(B0_long_rf<-randomForest(y=dt_b0$LONGITUDE, x=dt_b0[WAPS], 
#                                        importance=T,maximize=T,
#                                         method="rf", trControl=fitControl,
#                                         ntree=100, mtry=52,allowParalel=TRUE))
# 
# saveRDS(B0_long_rf, file = "./models/B0_long_rf.rds")

B0_long_rf<-readRDS("./models/B0_long_rf.rds")
Predictors_B0_long<-predict(B0_long_rf, dv_b0)
B0_postResample<-postResample(Predictors_B0_long, dv_b0$LONGITUDE_orig)
B0_postResample

error_rf<- dv_b0$LONGITUDE_orig- Predictors_B0_long

# KNN __________________________________________________________________________    
# system.time(B0_long_knn<-knnreg(LONGITUDE ~., data = dt_b0[,c(WAPS, "LONGITUDE")]))
# saveRDS(B0_long_knn, file = "./models/B0_long_knn.rds")

B0_long_knn<-readRDS("./models/B0_long_knn.rds")
Predictors_B0_long<-predict(B0_long_knn, dv_b0)
B0_postResample<-postResample(Predictors_B0_long, dv_b0$LONGITUDE_orig)
B0_postResample

error_knn<- dv_b0$LONGITUDE_orig- Predictors_B0_long


# SVM __________________________________________________________________________
# system.time(B0_long_svm <- svm(y = dt_b0$LONGITUDE, x=dt_b0[WAPS], kernel = "linear"))
# saveRDS(B0_long_svm, file = "./models/B0_long_svm.rds")

B0_long_svm<-readRDS("./models/B0_long_svm.rds")
Predictors_B0_long<-predict(B0_long_svm, dv_b0[WAPS])
B0_postResample<-postResample(Predictors_B0_long, dv_b0$LONGITUDE_orig)
B0_postResample

error_svm<- dv_b0$LONGITUDE_orig- Predictors_B0_long

# Plot the error distribution __________________________________________________
error<-cbind(error_knn, error_rf, error_svm)
error<-melt(error) %>% select(-Var1)
colnames(error)<-c("Model", "Value")
ggplot(error, aes(x=Value, fill=Model)) + 
  geom_histogram(alpha = 0.3, aes(y = ..density..), position = 'identity') +
  scale_fill_brewer(palette = "Dark2") +
  scale_x_continuous(breaks=seq(-50, 50, 10)) +
  facet_grid(~Model)

rm(error_rf, error_knn, error_svm,B0_long_rf,B0_long_knn, B0_long_svm, 
   Predictors_B0_long,B0_postResample, error)

##### D.2. Longitude building 1 #####
# Random forest ________________________________________________________________
# bestmtry_dt_b1<-tuneRF(dt_b1[WAPS], dt_b1$LONGITUDE, ntreeTry=100, stepFactor=2, 
#                        improve=0.05,trace=TRUE, plot=T)   # <- 103

# system.time(B1_long_rf<-randomForest(y=dt_b1$LONGITUDE, x=dt_b1[WAPS], 
#                                         importance=T,maximize=T,
#                                          method="rf", trControl=fitControl,
#                                          ntree=100, mtry=103,allowParalel=TRUE))
 
# saveRDS(B1_long_rf, file = "./models/B1_long_rf.rds")

B1_long_rf<-readRDS("./models/B1_long_rf.rds")
Predictors_b1_long<-predict(B1_long_rf, dv_b1)
b1_postResample<-postResample(Predictors_b1_long, dv_b1$LONGITUDE_orig)
b1_postResample

error_rf<- dv_b1$LONGITUDE_orig- Predictors_b1_long

# KNN __________________________________________________________________________    
# system.time(B1_long_knn<-knnreg(LONGITUDE ~., data = dt_b1[,c(WAPS, "LONGITUDE")]))
# saveRDS(B1_long_knn, file = "./models/B1_long_knn.rds")

B1_long_knn<-readRDS("./models/B1_long_knn.rds")
Predictors_b1_long<-predict(B1_long_knn, dv_b1)
b1_postResample<-postResample(Predictors_b1_long, dv_b1$LONGITUDE_orig)
b1_postResample

error_knn<- dv_b1$LONGITUDE_orig- Predictors_b1_long


# SVM __________________________________________________________________________
# system.time(B1_long_svm <- svm(y = dt_b1$LONGITUDE, x=dt_b1[WAPS], kernel = "linear"))
# saveRDS(B1_long_svm, file = "./models/B1_long_svm.rds")

B1_long_svm<-readRDS("./models/B1_long_svm.rds")
Predictors_b1_long<-predict(B1_long_svm, dv_b1[WAPS])
b1_postResample<-postResample(Predictors_b1_long, dv_b1$LONGITUDE_orig)
b1_postResample

error_svm<- dv_b1$LONGITUDE_orig- Predictors_b1_long

# Plot the error distribution __________________________________________________
error<-cbind(error_knn, error_rf, error_svm)
error<-melt(error) %>% select(-Var1)
colnames(error)<-c("Model", "Value")
ggplot(error, aes(x=Value, fill=Model)) + 
  geom_histogram(alpha = 0.3, aes(y = ..density..), position = 'identity') +
  scale_fill_brewer(palette = "Dark2") +
  scale_x_continuous(breaks=seq(-50, 50, 10)) +
  facet_grid(~Model)

rm(error_rf, error_knn, error_svm,B1_long_rf,B1_long_knn, B1_long_svm, 
   Predictors_b1_long,b1_postResample, error)

##### D.3. Longitude building 2 #####
# Random forest ________________________________________________________________
# bestmtry_dt_b2<-tuneRF(dt_b2[WAPS], dt_b2$LONGITUDE, ntreeTry=100, stepFactor=2, 
#                         improve=0.05,trace=TRUE, plot=T)   # <- 52

# system.time(B2_long_rf<-randomForest(y=dt_b2$LONGITUDE, x=dt_b2[WAPS], 
#                                          importance=T,maximize=T,
#                                           method="rf", trControl=fitControl,
#                                           ntree=100, mtry=52,allowParalel=TRUE))

# saveRDS(B2_long_rf, file = "./models/B2_long_rf.rds")

B2_long_rf<-readRDS("./models/B2_long_rf.rds")
Predictors_b2_long<-predict(B2_long_rf, dv_b2)
b2_postResample<-postResample(Predictors_b2_long, dv_b2$LONGITUDE_orig)
b2_postResample

error_rf<- dv_b2$LONGITUDE_orig- Predictors_b2_long

# KNN __________________________________________________________________________    
# system.time(B2_long_knn<-knnreg(LONGITUDE ~., data = dt_b2[,c(WAPS, "LONGITUDE")]))
# saveRDS(B2_long_knn, file = "./models/B2_long_knn.rds")

B2_long_knn<-readRDS("./models/B2_long_knn.rds")
Predictors_b2_long<-predict(B2_long_knn, dv_b2)
b2_postResample<-postResample(Predictors_b2_long, dv_b2$LONGITUDE_orig)
b2_postResample

error_knn<- dv_b2$LONGITUDE_orig- Predictors_b2_long


# SVM __________________________________________________________________________
# system.time(B2_long_svm <- svm(y = dt_b2$LONGITUDE, x=dt_b2[WAPS], kernel = "linear"))
# saveRDS(B2_long_svm, file = "./models/B2_long_svm.rds")

B2_long_svm<-readRDS("./models/B2_long_svm.rds")
Predictors_b2_long<-predict(B2_long_svm, dv_b2[WAPS])
b2_postResample<-postResample(Predictors_b2_long, dv_b2$LONGITUDE_orig)
b2_postResample

error_svm<- dv_b2$LONGITUDE_orig- Predictors_b2_long

# Plot the error distribution __________________________________________________
error<-cbind(error_knn, error_rf, error_svm)
error<-melt(error) %>% select(-Var1)
colnames(error)<-c("Model", "Value")
ggplot(error, aes(x=Value, fill=Model)) + 
  geom_histogram(alpha = 0.3, aes(y = ..density..), position = 'identity') +
  scale_fill_brewer(palette = "Dark2") +
  scale_x_continuous(breaks=seq(-50, 50, 10)) +
  facet_grid(~Model)

rm(error_rf, error_knn, error_svm,B2_long_rf,B2_long_knn, B2_long_svm, 
   Predictors_b2_long,b2_postResample, error)

#### E.PREDICTING LATITUDE PER BUILDING (1 model per building) ####
WAPS<- intersect(grep("WAP",names(dt_b2),value=T),
                 grep("High",names(dt_b2),invert=TRUE,value=T))
##### E.1. Latitude building 0 ####
# Random forest ________________________________________________________________
# bestmtry_dt_b0<-tuneRF(dt_b0[WAPS], dt_b0$LATITUDE, ntreeTry=100, stepFactor=2, 
#                        improve=0.05,trace=TRUE, plot=T)   # <- 52

# system.time(B0_lat_rf<-randomForest(y=dt_b0$LATITUDE, x=dt_b0[WAPS], 
#                                         importance=T,maximize=T,
#                                          method="rf", trControl=fitControl,
#                                          ntree=100, mtry=52,allowParalel=TRUE))
#  
# saveRDS(B0_lat_rf, file = "./models/B0_lat_rf.rds")

B0_lat_rf<-readRDS("./models/B0_lat_rf.rds")
Predictors_B0_lat<-predict(B0_lat_rf, dv_b0)
B0_postResample<-postResample(Predictors_B0_lat, dv_b0$LATITUDE_orig)
B0_postResample

error_rf<- dv_b0$LATITUDE_orig- Predictors_B0_lat

# KNN __________________________________________________________________________    
# system.time(B0_lat_knn<-knnreg(LATITUDE ~., data = dt_b0[,c(WAPS, "LATITUDE")]))
# saveRDS(B0_lat_knn, file = "./models/B0_lat_knn.rds")

B0_lat_knn<-readRDS("./models/B0_lat_knn.rds")
Predictors_B0_lat<-predict(B0_lat_knn, dv_b0)
B0_postResample<-postResample(Predictors_B0_lat, dv_b0$LATITUDE_orig)
B0_postResample

error_knn<- dv_b0$LATITUDE_orig- Predictors_B0_lat

# SVM __________________________________________________________________________
# system.time(B0_lat_svm <- svm(y = dt_b0$LATITUDE, x=dt_b0[WAPS], kernel = "linear"))
# saveRDS(B0_lat_svm, file = "./models/B0_lat_svm.rds")

B0_lat_svm<-readRDS("./models/B0_lat_svm.rds")
Predictors_B0_lat<-predict(B0_lat_svm, dv_b0[WAPS])
B0_postResample<-postResample(Predictors_B0_lat, dv_b0$LATITUDE_orig)
B0_postResample

error_svm<- dv_b0$LATITUDE_orig- Predictors_B0_lat

# Plot the error distribution __________________________________________________
error<-cbind(error_knn, error_rf, error_svm)
error<-melt(error) %>% select(-Var1)
colnames(error)<-c("Model", "Value")
ggplot(error, aes(x=Value, fill=Model)) + 
  geom_histogram(alpha = 0.3, aes(y = ..density..), position = 'identity') +
  scale_fill_brewer(palette = "Dark2") +
  scale_x_continuous(breaks=seq(-50, 50, 10)) +
  facet_grid(~Model)

rm(error_rf, error_knn, error_svm,B0_lat_rf,B0_lat_knn, B0_lat_svm, 
   Predictors_B0_lat,B0_postResample, error)

##### E.2. Latitude building 1 ####
# Random forest ________________________________________________________________
# bestmtry_dt_b1<-tuneRF(dt_b1[WAPS], dt_b1$LATITUDE, ntreeTry=100, stepFactor=2, 
#                         improve=0.05,trace=TRUE, plot=T)   # <- 103

# system.time(B1_lat_rf<-randomForest(y=dt_b1$LATITUDE, x=dt_b1[WAPS], 
#                                          importance=T,maximize=T,
#                                           method="rf", trControl=fitControl,
#                                           ntree=100, mtry=103,allowParalel=TRUE))
  
# saveRDS(B1_lat_rf, file = "./models/B1_lat_rf.rds")

B1_lat_rf<-readRDS("./models/B1_lat_rf.rds")
Predictors_b1_lat<-predict(B1_lat_rf, dv_b1)
b1_postResample<-postResample(Predictors_b1_lat, dv_b1$LATITUDE_orig)
b1_postResample

error_rf<- dv_b1$LATITUDE_orig- Predictors_b1_lat

# KNN __________________________________________________________________________    
# system.time(B1_lat_knn<-knnreg(LATITUDE ~., data = dt_b1[,c(WAPS, "LATITUDE")]))
# saveRDS(B1_lat_knn, file = "./models/B1_lat_knn.rds")

B1_lat_knn<-readRDS("./models/B1_lat_knn.rds")
Predictors_b1_lat<-predict(B1_lat_knn, dv_b1)
b1_postResample<-postResample(Predictors_b1_lat, dv_b1$LATITUDE_orig)
b1_postResample

error_knn<- dv_b1$LATITUDE_orig- Predictors_b1_lat

# SVM __________________________________________________________________________
# system.time(B1_lat_svm <- svm(y = dt_b1$LATITUDE, x=dt_b1[WAPS], kernel = "linear"))
# saveRDS(B1_lat_svm, file = "./models/B1_lat_svm.rds")

B1_lat_svm<-readRDS("./models/B1_lat_svm.rds")
Predictors_b1_lat<-predict(B1_lat_svm, dv_b1[WAPS])
b1_postResample<-postResample(Predictors_b1_lat, dv_b1$LATITUDE_orig)
b1_postResample

error_svm<- dv_b1$LATITUDE_orig- Predictors_b1_lat

# Plot the error distribution __________________________________________________
error<-cbind(error_knn, error_rf, error_svm)
error<-melt(error) %>% select(-Var1)
colnames(error)<-c("Model", "Value")
ggplot(error, aes(x=Value, fill=Model)) + 
  geom_histogram(alpha = 0.3, aes(y = ..density..), position = 'identity') +
  scale_fill_brewer(palette = "Dark2") +
  scale_x_continuous(breaks=seq(-50, 50, 10)) +
  facet_grid(~Model)

rm(error_rf, error_knn, error_svm,B1_lat_rf,B1_lat_knn, B1_lat_svm, 
   Predictors_b1_lat,b1_postResample, error)

##### E.3. Latitude building 2 ####

# Random forest ________________________________________________________________
# bestmtry_dt_b2<-tuneRF(dt_b2[WAPS], dt_b2$LATITUDE, ntreeTry=100, stepFactor=2, 
#                       improve=0.05,trace=TRUE, plot=T)   # <- 52

# system.time(B2_lat_rf<-randomForest(y=dt_b2$LATITUDE, x=dt_b2[WAPS], 
#                                          importance=T,maximize=T,
#                                           method="rf", trControl=fitControl,
#                                           ntree=100, mtry=52,allowParalel=TRUE))
   
# saveRDS(B2_lat_rf, file = "./models/B2_lat_rf.rds")

B2_lat_rf<-readRDS("./models/B2_lat_rf.rds")
Predictors_b2_lat<-predict(B2_lat_rf, dv_b2)
b2_postResample<-postResample(Predictors_b2_lat, dv_b2$LATITUDE_orig)
b2_postResample

error_rf<- dv_b2$LATITUDE_orig- Predictors_b2_lat

# KNN __________________________________________________________________________    
# system.time(B2_lat_knn<-knnreg(LATITUDE ~., data = dt_b2[,c(WAPS, "LATITUDE")]))
# saveRDS(B2_lat_knn, file = "./models/B2_lat_knn.rds")

B2_lat_knn<-readRDS("./models/B2_lat_knn.rds")
Predictors_b2_lat<-predict(B2_lat_knn, dv_b2)
b2_postResample<-postResample(Predictors_b2_lat, dv_b2$LATITUDE_orig)
b2_postResample

error_knn<- dv_b2$LATITUDE_orig- Predictors_b2_lat

# SVM __________________________________________________________________________
# system.time(B2_lat_svm <- svm(y = dt_b2$LATITUDE, x=dt_b2[WAPS], kernel = "linear"))
# saveRDS(B2_lat_svm, file = "./models/B2_lat_svm.rds")

B2_lat_svm<-readRDS("./models/B2_lat_svm.rds")
Predictors_b2_lat<-predict(B2_lat_svm, dv_b2[WAPS])
b2_postResample<-postResample(Predictors_b2_lat, dv_b2$LATITUDE_orig)
b2_postResample

error_svm<- dv_b2$LATITUDE_orig- Predictors_b2_lat

# Plot the error distribution __________________________________________________
error<-cbind(error_knn, error_rf, error_svm)
error<-melt(error) %>% select(-Var1)
colnames(error)<-c("Model", "Value")
ggplot(error, aes(x=Value, fill=Model)) + 
  geom_histogram(alpha = 0.3, aes(y = ..density..), position = 'identity') +
  scale_fill_brewer(palette = "Dark2") +
  scale_x_continuous(breaks=seq(-50, 50, 10)) +
  facet_grid(~Model)

rm(error_rf, error_knn, error_svm,B2_lat_rf,B2_lat_knn, B2_lat_svm, 
   Predictors_b2_lat,b2_postResample, error)



#### 16. FINAL VISUALIZATION ####
VarInd<-c("LATITUDE", "LONGITUDE", "FLOOR")
VisLatLon<- gdata::combine(DataValidOrig[VarInd], DataValid[VarInd]) 
VisLatLon$source<-ifelse(VisLatLon$source=="DataValidOrig[VarInd]", "Original", "Predicted")
VisLatLon$source<-as.factor(VisLatLon$source)

p <- plot_ly(VisLatLon, x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, color = ~source,
             colors=c("grey","blue")) %>%
  add_markers() %>%
  
  layout(scene = list(xaxis = list(title = 'Longitude'),
                      yaxis = list(title = 'Latitude'),
                      zaxis = list(title = 'Floor')))

ggplot(DataValidOrig, aes(x = LONGITUDE, y = LATITUDE)) + 
  geom_point (stat="identity", position=position_dodge(),color="grey53") + 
  labs(x = "Longitude", y = "Latitude", title = "Predictions per Floor")  +
  facet_wrap(~FLOOR) + 
  geom_point(data = DataValid, aes(x = LONGITUDE, y = LATITUDE), color="blue") 

plot(LATITUDE ~ LONGITUDE, data = DataValidOrig, pch = 20, col = "grey53")
points(LATITUDE ~ LONGITUDE, data = DataValid, pch = 20, col = "blue")

stopCluster(cluster)
rm(cluster)
registerDoSEQ()



