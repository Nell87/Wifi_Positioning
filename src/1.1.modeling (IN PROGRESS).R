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

##### D.PREDICTING LONGITUDE PER BUILDING (1 model per building) ##### 
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

saveRDS(B2_long_rf, file = "./models/B2_long_rf.rds")

B2_long_rf<-readRDS("./models/B2_long_rf.rds")
Predictors_b2_long<-predict(B2_long_rf, dv_b2)
b2_postResample<-postResample(Predictors_b2_long, dv_b2$LONGITUDE_orig)
b2_postResample

error_rf<- dv_b2$LONGITUDE_orig- Predictors_b2_long

# KNN __________________________________________________________________________    
system.time(B2_long_knn<-knnreg(LONGITUDE ~., data = dt_b2[,c(WAPS, "LONGITUDE")]))
saveRDS(B2_long_knn, file = "./models/B2_long_knn.rds")

B2_long_knn<-readRDS("./models/B2_long_knn.rds")
Predictors_b2_long<-predict(B2_long_knn, dv_b2)
b2_postResample<-postResample(Predictors_b2_long, dv_b2$LONGITUDE_orig)
b2_postResample

error_knn<- dv_b2$LONGITUDE_orig- Predictors_b2_long


# SVM __________________________________________________________________________
system.time(B2_long_svm <- svm(y = dt_b2$LONGITUDE, x=dt_b2[WAPS], kernel = "linear"))
saveRDS(B2_long_svm, file = "./models/B2_long_svm.rds")

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

rm(error_rf, error_knn, error_svm,b2_long_rf,b2_long_knn, b2_long_svm, 
   Predictors_b2_long,b2_postResample, error)

##### E.PREDICTING LATITUDE PER BUILDING (1 model per building) ##### 
##### E.1. Latitude building 0
##### E.2. Latitude building 1
##### E.3. Latitude building 2













# Random Forest -B0-    <- # RMSE 7.01  R 0.932  MAE 4.64 TIME 52   
# WAPS
WAPS_B0<-intersect(grep("WAP",names(BUILDING0),value=T),
                   grep("High",names(BUILDING0),invert=TRUE,value=T))

# bestmtry_LON_BO_RF<-tuneRF(BUILDING0[WAPS_B0], BUILDING0$LONGITUDE, ntreeTry=100, stepFactor=2, 
#                       improve=0.05,trace=TRUE, plot=T)  

# system.time(LON_BO_RF<-randomForest(y=BUILDING0$LONGITUDE, x=BUILDING0[WAPS_B0], 
#                                      importance=T,maximize=T,
#                                       method="rf", trControl=fitControl,
#                                       ntree=100, mtry=52,allowParalel=TRUE))
  
    
# save(LON_BO_RF, file = "./models/LON_BO_RF.rda")
load("./models/LON_BO_RF.rda")
predictions_LONBORF<-predict(LON_BO_RF, df_datavalid[df_datavalid$BUILDINGID==0, ])
rf_postRes_LONBORF<-postResample(predictions_LONBORF, 
                                 df_datavalid$LONGITUDE[df_datavalid$BUILDINGID==0])
rf_postRes_LONBORF

# Random Forest -B1-        <- # RMSE 9.21  R 0.96  MAE 6.59   TIME 42 sec
# bestmtry_LON_B1_RF<-tuneRF(BUILDING1[WAPS], BUILDING1$LONGITUDE, ntreeTry=100, stepFactor=2, 
#                            improve=0.05,trace=TRUE, plot=T)

# system.time(LON_B1_RF<-randomForest(LONGITUDE~. -LATITUDE -FLOOR -SPACEID -RELATIVEPOSITION -USERID 
#                                     -PHONEID -TIMESTAMP -source -HighWAP -HighRSSI -Build_floorID -ID, 
#                                      data= BUILDING1, 
#                                      importance=T,maximize=T,
#                                      method="rf", trControl=fitControl,
#                                      ntree=100, mtry=104,allowParalel=TRUE))
# save(LON_B1_RF, file = "LON_B1_RF.rda")

load("LON_B1_RF.rda")
predictions_LONB1RF<-predict(LON_B1_RF, DataValid[DataValid$BUILDINGID==1, ])
rf_postRes_LONB1RF<-postResample(predictions_LONB1RF, DataValidOrig$LONGITUDE[DataValidOrig$BUILDINGID==1])
rf_postRes_LONB1RF

# Random Forest -B2-       <- # RMSE 10.43  R 0.89   MAE 7.14  TIME  sec 86
# bestmtry_LON_B2_RF<-tuneRF(BUILDING2[WAPS], BUILDING2$LONGITUDE, ntreeTry=100, stepFactor=2, 
#                            improve=0.05,trace=TRUE, plot=T)

# system.time(LON_B2_RF<-randomForest(LONGITUDE~. -LATITUDE -FLOOR -SPACEID -RELATIVEPOSITION -USERID 
#                                     -PHONEID -TIMESTAMP -source -HighWAP -HighRSSI -Build_floorID -ID,
#                                      data= BUILDING2, 
#                                      importance=T,maximize=T,
#                                      method="rf", trControl=fitControl,
#                                      ntree=100, mtry=52,allowParalel=TRUE))
# save(LON_B2_RF, file = "LON_B2_RF.rda")

load("LON_B2_RF.rda")
predictions_LONB2RF<-predict(LON_B2_RF, DataValid[DataValid$BUILDINGID==2, ])
rf_postRes_LONB2RF<-postResample(predictions_LONB2RF, DataValidOrig$LONGITUDE[DataValidOrig$BUILDINGID==2])
rf_postRes_LONB2RF

#### 8. PREDICTING LONGITUDE USING PCA ####
# Add dummy variable for BuildingID 
DummyVar <- dummyVars("~BUILDINGID", data = Data_Full, fullRank=T)
DummyVarDF <- data.frame(predict(DummyVar, newdata = Data_Full))
Data_Full<-cbind(Data_Full, DummyVarDF)

#store WAP names in a vector. Exclude those variable names that do not have at least one number
#(i.e. "max_WAP")

WAPs<-grep("[[:digit:]]",(grep("WAP", names(Data_Full), value=T)), value=T)#312 WAPs

#PC overview with the caret package
pca1<-preProcess(Data_Full[Data_Full$source=="DataTrain", WAPs], 
                 #78 components capture 80 percent of the variance
                 method=c("center", "scale", "pca"), thresh=0.8)

#78 components capture 80 percent of the variance

#singular value decomposition
# By default, prcomp already centers the variable to have mean equals to zero

training_WAPs<-Data_Full[Data_Full$source=="DataTrain", WAPs]
nrow(training_WAPs)#19,227 obs

pca2 <- prcomp(training_WAPs, scale. = T)
names(pca2)
pca2$rotation #312 PC loadings

#plot the 78 resulting pc's. Scale = 0 ensures that arrows are scaled to represent the loadings
biplot(pca2[,1:78], scale = 0)#can't see anything!

std_dev <- pca2$sdev
var <- std_dev^2 #eigenvalues
prop_var <- var/sum(var)

summary(pca2)

#Decide how many PC to keep for the modeling stage: use scree plot.

#scree plot
plot(prop_var, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

#cumulative scree plot
plot(cumsum(prop_var), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

sum(prop_var[1:78])

# loadings (eigenvectors)
# head(pca2$rotation)

# PCs (aka scores)
# head(pca2$x)

#From the graph above we can infer that 78 components capture 80 pct
#of the variance, as already determined in "pca1".
autoplot(pca2, scale=0)

#Change training set WAPs for the 78 principal components
WAPs<-grep("[[:digit:]]",(grep("WAP", names(Data_Full), value=T)), value=T)
train_pca <- data.frame(Data_Full[Data_Full$source=="DataTrain", -which(colnames(Data_Full) %in% WAPs)],
                        pca2$x[,1:78])
nrow(train_pca)#19,227

#We are interested in the first 78 PCs
nrow(pca2$x)
nrow(train_pca)#19,227 obs

valid_pca <- predict(pca2, newdata = Data_Full[Data_Full$source=="DataValid",])
#select the first 78 components
valid_pca <- valid_pca[, 1:78]

valid_pca <- data.frame(Data_Full[Data_Full$source=="DataValid", -which(colnames(Data_Full) %in% WAPs)],
                        valid_pca)
all.equal(colnames(train_pca), colnames(valid_pca))#TRUE

##### 8.1. Random Forest #####     RMSE 8.92  R 0.994  MAE  6.04  TIME 183 sec  
# system.time(LON_PCA<-randomForest(LONGITUDE~. -LATITUDE -FLOOR -BUILDINGID -SPACEID -RELATIVEPOSITION
#                                   -USERID-PHONEID-TIMESTAMP-source-ID -HighWAP -HighRSSI -Build_floorID,
#                                   data=train_pca,
#                                   method="rf",
#                                   ntree=100,
#                                   trControl=fitControl,
#                                   importance = T,
#                                   maximize =T, 
#                                   allowParalel=TRUE))
# save(LON_PCA, file = "LON_PCA.rda")

load("LON_PCA.rda")
predictions_LON_PCA<-predict(LON_PCA, valid_pca)

rf_postRes_LON_PCA<-postResample(predictions_LON_PCA, valid_pca$LONGITUDE)
rf_postRes_LON_PCA

#8.9258367 0.9945811 6.0411906

#### 9. PREDICTING LONGITUDE IN ALL BUILDINGS (the same model for every building) ####------------------
# Split Data before modeling
Data_FullSplit<-split(Data_Full, Data_Full$source)
list2env(Data_FullSplit, envir=.GlobalEnv)
rm(Data_FullSplit)

##### 9.1. Random Forest #####   RMSE 8.44  R 0.995  MAE  5.66  TIME 712 sec  
# VarInd<-c(WAPS, "BUILDINGID.1", "BUILDINGID.2")
# bestmtry<-tuneRF(DataTrain[VarInd], DataTrain$LONGITUDE, ntreeTry=100, stepFactor=2, improve=0.05,  
#                   trace=TRUE, plot=T)

# system.time(LON_AllB_RF<-randomForest(LONGITUDE~. -LATITUDE -FLOOR -SPACEID -RELATIVEPOSITION -USERID -PHONEID 
#                                      -TIMESTAMP -source -HighWAP -HighRSSI -Build_floorID -BUILDINGID -ID, 
#                                      data= DataTrain, 
#                                      importance=T,maximize=T,
#                                      method="rf", trControl=fitControl,
#                                      ntree=100, mtry= 104, allowParalel=TRUE))
# save(LON_AllB_RF, file = "LON_AllB_RF.rda")

load("LON_AllB_RF.rda")
predictions_LON_AllBRF<-predict(LON_AllB_RF, DataValid)
rf_postRes_LON_AllBRF<-postResample(predictions_LON_AllBRF, DataValid$LONGITUDE)
rf_postRes_LON_AllBRF

# Visualizing errors
Problems_LON<-as.data.frame(cbind(predictions_LON_AllBRF, DataValidOrig$LONGITUDE))
Problems_LON<-Problems_LON %>% mutate(Error= Problems_LON[,1]-Problems_LON[,2]) %>%
  mutate(ID=DataValid$ID)
Problems_LON10m<-Problems_LON %>% filter(abs(Error)>10)
boxplot(Problems_LON10m$Error)
hist(Problems_LON$Error, xlab = "Error", ylab="Frequency", 
     main="Error predicting Longitude")

boxplot(Problems_LON10m$Error, xlab = "Error", ylab="Frequency", 
        main="Error predicting Longitude >10m")

# Mean Error
MeanErrorLON<-mean(abs(Problems_LON$Error))     # mean: 5.66m   median:3.84m

# Subset the sample with these errors
Indices10m<-Problems_LON10m$ID
DF_LON10m<-DataValidOrig[DataValidOrig$ID %in% Indices10m, ]
DF_LON10m_NOWAPS<-DF_LON10m %>% select(313:326)

# Plot long & lat
ggplot(DataValidOrig, aes(x = LONGITUDE, y = LATITUDE)) + 
  geom_point (stat="identity", position=position_dodge(),color="grey53") + 
  labs(x = "Longitude", y = "Latitude", title = "Errors >10m")  +
  facet_wrap(~FLOOR) + 
  geom_point(data = DF_LON10m, aes(x = LONGITUDE, y = LATITUDE), color="red") 

plot(LATITUDE ~ LONGITUDE, data = DataValidOrig, pch = 20, col = "grey53")
points(LATITUDE ~ LONGITUDE, data = DF_LON10m, pch = 20, col = "red")

##### 9.2. KNN #####      RMSE 20.23  R 0.972  MAE  8.04  TIME 687 sec 
# Scale Data: Calculate the pre-process parameters
# preprocessParams <- preProcess(Data_Full[,c(WAPS, "LATITUDE")], method=c("scale", "center"))
# 
# # Transform the dataset using the parameters
# Data_Full_KNN <- predict(preprocessParams, Data_Full)
# 
# # Split data
# Data_FullSplit<-split(Data_Full_KNN, Data_Full_KNN$source)
# names(Data_FullSplit)<-c("DataTrainKNN", "DataValidKNN")
# list2env(Data_FullSplit, envir=.GlobalEnv)
# rm(Data_FullSplit)

# system.time(LON_AllB_KNN<-caret::train(LONGITUDE~. -LATITUDE -FLOOR -BUILDINGID -SPACEID -RELATIVEPOSITION
#                                  -USERID-PHONEID-TIMESTAMP-source-ID -HighWAP -HighRSSI -Build_floorID, 
#                                  data=DataTrainKNN, 
#                                  method="knn"))
# 
# save(LON_AllB_KNN, file = "LON_AllB_KNN.rda")

# load("LON_AllB_KNN.rda")
# predictions_LON_AllBKNN<-predict(LON_AllB_KNN, DataValid)
# rf_postRes_LON_AllBKNN<-postResample(predictions_LON_AllBKNN, DataValid$LONGITUDE)
# rf_postRes_LON_AllBKNN
# 
# 
# ##### 9.3. SVM #####    
# ParamSVM <- tune(svm, DataTrainKNN$LONGITUDE~., data = DataTrainKNN[,c(WAPS, "BUILDINGID.1","BUILDINGID.2")],
#             ranges = list(gamma = 2^(-1:1), cost = 2^(2:4))) 

#system.time(LON_AllB_SVM<-caret::train(LONGITUDE~. -LATITUDE -FLOOR -BUILDINGID -SPACEID -RELATIVEPOSITION
#                                         -USERID-PHONEID-TIMESTAMP-source-ID -HighWAP -HighRSSI -Build_floorID, 
#                                         data=DataTrain, 
#                                         method="svmLinear",
#                                         preProcess=c("center", "scale"), 
#                                         trControl=fitControl))
#  
# save(LON_AllB_SVM, file = "LON_AllB_SVM.rda")
# 
# load("LON_AllB_SVM.rda")
# predictions_LON_AllBSVM<-predict(LON_AllB_SVM, DataValid)
# rf_postRes_LON_AllBSVM<-postResample(predictions_LON_AllBSVM, DataValid$LONGITUDE)
# rf_postRes_LON_AllBSVM

#### 10. PREDICTING LATITUDE USING PCA ####----------------------------------------------------------------------
##### 10.1. Random Forest #####     RMSE 8.35  R 0.986  MAE  5.57  TIME 194 sec  
# system.time(LAT_PCA<-randomForest(LATITUDE~. -LONGITUDE -FLOOR -BUILDINGID -SPACEID -RELATIVEPOSITION
#                                    -USERID-PHONEID-TIMESTAMP-source-ID -HighWAP -HighRSSI -Build_floorID,
#                                    data=train_pca,
#                                    method="rf",
#                                    ntree=100,
#                                    trControl=fitControl,
#                                    importance = T,
#                                    maximize =T, 
#                                   allowParalel=TRUE))
# save(LAT_PCA, file = "LAT_PCA.rda")

load("LAT_PCA.rda")
predictions_LAT_PCA<-predict(LAT_PCA, valid_pca)

rf_postRes_LAT_PCA<-postResample(predictions_LAT_PCA, valid_pca$LATITUDE)
rf_postRes_LAT_PCA

#8.3538246 0.9860792 5.5691548

#### 11. PREDICTING LATITUDE IN ALL BUILDINGS (the same model for every building) ####------------------
##### 11.1. Random Forest #####    <- # RMSE 8.11  R 0.986  MAE 5.47  TIME 646 sec  
#VarInd<-c(WAPS, "BUILDINGID.1", "BUILDINGID.2")
#bestmtry<-tuneRF(DataTrain[VarInd], DataTrain$LATITUDE, ntreeTry=100, stepFactor=2, improve=0.05,  
#                   trace=TRUE, plot=T)

#system.time(LAT_AllB_RF<-randomForest(LATITUDE~. -LONGITUDE -FLOOR -SPACEID -RELATIVEPOSITION -USERID -PHONEID 
#                                       -TIMESTAMP -source -HighWAP -HighRSSI -Build_floorID -BUILDINGID -ID, 
#                                       data= DataTrain, 
#                                       importance=T,maximize=T,
#                                       method="rf", trControl=fitControl,
#                                       ntree=100, mtry= 104,allowParalel=TRUE))
# save(LAT_AllB_RF, file = "LAT_AllB_RF.rda")

load("LAT_AllB_RF.rda")
predictions_LAT_AllBRF<-predict(LAT_AllB_RF, DataValid)
rf_postRes_LAT_AllBRF<-postResample(predictions_LAT_AllBRF, DataValid$LATITUDE)
rf_postRes_LAT_AllBRF

# Visualizing errors
Problems_LAT<-as.data.frame(cbind(predictions_LAT_AllBRF, DataValidOrig$LATITUDE))
Problems_LAT<-Problems_LAT %>% mutate(Error= Problems_LAT[,1]-Problems_LAT[,2]) %>%
  mutate(ID=DataValid$ID)
Problems_LAT10m<-Problems_LAT %>% filter(abs(Error)>10)

hist(Problems_LAT$Error, xlab = "Error", ylab="Frequency", 
     main="Error predicting Latitude" )

boxplot(Problems_LAT10m$Error, xlab = "Error", ylab="Frequency", 
        main="Error predicting Latitude >10m" )

# Mean Error
MeanErrorLAT<-mean(abs(Problems_LAT$Error))     # mean:5.47m  median: 3.54m

# Subset the sample with these errors
Indices10m<-Problems_LAT$ID
DF_LAT10m<-DataValidOrig[DataValidOrig$ID %in% Indices10m, ]

##### 11.2. KNN #####    
# system.time(LAT_AllB_KNN<-caret::train(LATITUDE~. -LONGITUDE -FLOOR -BUILDINGID -SPACEID -RELATIVEPOSITION
#                                        -USERID-PHONEID-TIMESTAMP-source-ID -HighWAP -HighRSSI -Build_floorID, 
#                                        data=DataTrain, 
#                                        method="knn",
#                                        preProcess=c("center", "scale"), 
#                                        trControl=fitControl))
# 
# save(LAT_AllB_KNN, file = "LAT_AllB_KNN.rda")

# load("LAT_AllB_KNN.rda")
# predictions_LAT_AllBKNN<-predict(LAT_AllB_KNN, DataValid)
# rf_postRes_LAT_AllBKNN<-postResample(predictions_LAT_AllBKNN, DataValid$LATGITUDE)
# rf_postRes_LAT_AllBKNN


##### 11.3. SVM #####    
# system.time(LAT_AllB_SVM<-caret::train(LATITUDE~. -LONGITUDE -FLOOR -BUILDINGID -SPACEID -RELATIVEPOSITION
#                                        -USERID-PHONEID-TIMESTAMP-source-ID -HighWAP -HighRSSI -Build_floorID, 
#                                        data=DataTrain, 
#                                        method="svmLinear",
#                                        preProcess=c("center", "scale"), 
#                                        trControl=fitControl))
# 
# save(LAT_AllB_SVM, file = "LAT_AllB_SVM.rda")

# load("LAT_AllB_SVM.rda")
# predictions_LAT_AllBSVM<-predict(LAT_AllB_SVM, DataValid)
# rf_postRes_LAT_AllBSVM<-postResample(predictions_LAT_AllBSVM, DataValid$LATGITUDE)
# rf_postRes_LAT_AllBSVM



#### 16. FINAL VISUALIZATION ####
# Add predicted floor
DataValid$FLOOR<-predictions_Floor_BuilLatLongRF

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



