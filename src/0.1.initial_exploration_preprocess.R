
#### 0. INCLUDES  --------------------------------------------------------------------

#Load Libraries: p_load can install, load,  and update packages
if(require("pacman")=="FALSE"){
  install.packages("pacman")
} 

pacman::p_load(dplyr, lubridate, rgl, reshape2,cowplot)

# Setwd
setwd("C:/SARA/PORTFOLIO/Wifi_Positioning/data/")

# Load Data
df_datatrain <- read.csv2("trainingData.csv", header=TRUE, sep=",",  stringsAsFactors=FALSE, na.strings=c("NA", "-", "?"))
df_datavalid<-read.csv2("validationData.csv", header=TRUE, sep=",",  stringsAsFactors=FALSE, na.strings=c("NA", "-", "?"))

##### A. FIRST CHECKS ---------------------------------------------------------
# Dimension Variables 
dim(df_datatrain)    # <- 19,937 x 529    <- We have the same number of variables
dim(df_datavalid)    # <- 1111   x 529

# Check if we have the same names of variables      #<-Yes!!!
"%ni%" <- Negate("%in%")
names(df_datatrain[which(names(df_datatrain) %ni% names(df_datavalid))])   #<-0
names(df_datavalid[which(names(df_datavalid) %ni% names(df_datatrain))])   #<-0

names(df_datatrain)  # <- 0:520 WAPS     521 LONGITUDE 522 LATITUDE          523 FLOOR     
#                      <- 524 BUILDINGID 525  SPACEID  526 RELATIVEPOSITION  527 USERID
#                      <- 528 PHONEID    529 TIMESTAMP

# Remove repeated rows in df_datatrain             
df_datatrain<-distinct(df_datatrain)              #<- 19937 to 19300

# Remove repeated rows in df_datavalidation          
df_datavalid<-distinct(df_datavalid)              #<- No repeated rows!!

# Missing Values  
sum(is.na(df_datatrain))   #<-0  No missing values!!
sum(is.na(df_datavalid))   #<-0  No missing values!!

##### B. TRANSFORMATIONS ----------------------------------------------------------
# Combine datasets for speeding-up the transformations and add ID
data_full<-gdata::combine(df_datatrain, df_datavalid)   #<-Combine adding the source in a new column
data_full<-data_full %>% mutate(id = row_number())

# Transform some variables to factor/numeric/datetime
factors<-c("FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "source")
data_full[,factors]<-lapply(data_full[,factors], as.factor)
rm(factors)

numeric<-c("LONGITUDE", "LATITUDE")
data_full[,numeric]<-lapply(data_full[,numeric], as.numeric)
rm(numeric)

#Let's compare locations
df_datatrain<-df_datatrain %>% mutate(Position=group_indices(df_datatrain, LATITUDE, LONGITUDE, FLOOR))
df_datavalid<-df_datavalid %>% mutate(Position=group_indices(df_datavalid, LATITUDE, LONGITUDE, FLOOR))

max(df_datatrain$Position)    # <- We have different positions in Train and Valid!!  O.O!!
max(df_datavalid$Position)
max(data_full$Position)
