
#### 0. INCLUDES  --------------------------------------------------------------------

#Load Libraries: p_load can install, load,  and update packages
if(require("pacman")=="FALSE"){
  install.packages("pacman")
} 

pacman::p_load(dplyr, lubridate, caret)

# Setwd
setwd("C:/SARA/PORTFOLIO/Wifi_Positioning/data/")

# Load Data
df_datatrain <- read.csv2("trainingData.csv", header=TRUE, sep=",",  stringsAsFactors=FALSE, na.strings=c("NA", "-", "?"))
df_datavalid<-read.csv2("validationData.csv", header=TRUE, sep=",",  stringsAsFactors=FALSE, na.strings=c("NA", "-", "?"))

#### A. FIRST CHECKS ---------------------------------------------------------
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

#### B. TRANSFORMATIONS ----------------------------------------------------------
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

data_full$TIMESTAMP <- as_datetime(data_full$TIMESTAMP,
                                   origin = "1970-01-01", tz="UTC")

# Let's save the variables in two vectors
WAPS<-grep("WAP", names(data_full), value=T)
NOWAPS<-names(df_datatrain[names(df_datatrain) %ni% WAPS])


#### C. EXPLORATION WAPS ----------------------------------------------------------
# Change value of WAPS= 100 to WAPS=-110 __________________________
data_full[,WAPS] <- sapply(data_full[,WAPS],function(x) ifelse(x==100,-110,x))

# Remove rows with no variance ____________________________________
# New WAPS
WAPS<-grep("WAP", names(data_full), value=T)

# Filter Rows with all RSSI = -110        <- From 20411 to 20338 row
data_full <- data_full %>% 
  filter(apply(data_full[WAPS], 1, function(x)length(unique(x)))>1)

# Select Relevant WAPS  ___________________________________________
WAPS_VarTrain<-nearZeroVar(data_full[data_full$source=="df_datatrain",WAPS], saveMetrics=TRUE)
WAPS_VarValid<-nearZeroVar(data_full[data_full$source=="df_datavalid",WAPS], saveMetrics=TRUE)

data_full<-data_full[-which(WAPS_VarTrain$zeroVar==TRUE | 
                              WAPS_VarValid$zeroVar==TRUE)]   # 531 -> 323 variables

rm(WAPS_VarTrain, WAPS_VarValid)

# New WAPS
WAPS<-grep("WAP", names(data_full), value=T)

# Create new variables HighWAP and HighRSSI _______________________________________
data_full<-data_full %>% 
  mutate(HighWAP=NA, HighRSSI=NA)

data_full<-data_full %>% 
  mutate(HighWAP=colnames(data_full[WAPS])[apply(data_full[WAPS],1,which.max)])

data_full<-data_full %>% 
  mutate(HighRSSI=apply(data_full[WAPS], 1, max))

# Transform to factor
data_full$HighWAP<-as.factor(data_full$HighWAP)
data_full$BUILDINGID<-as.factor(data_full$BUILDINGID)

# Remove from "max_WAP" those WAPs that never provided the best signal to any of the observations
# of the training set 
outersect <- function(x, y) {
  x[!x%in%y]
}

remove_WAPs<-unique(
  outersect(data_full$HighWAP[data_full$source=="df_datavalid"],
            data_full$HighWAP[data_full$source=="df_datatrain"])
)

remove_WAPs

# Remove WAPs 268 & 323
data_full<-data_full[-which(data_full$HighWAP %in% remove_WAPs),]

# New WAPS
WAPS<-grep("WAP", names(data_full), value=T)

# Are there same HighWAP in different Buildings? _______________________________________
WAPS_Recoloc<-data_full %>%
  select(HighWAP, BUILDINGID, source) %>%
  distinct(HighWAP, BUILDINGID, source)

RepWAPS<-WAPS_Recoloc %>% distinct(HighWAP, BUILDINGID)
RepWAPS<-RepWAPS %>% group_by(HighWAP) %>% summarise(count=n())
                    # WAP 248 is highwap in the 3 buildings!!
rm(RepWAPS, WAPS_Recoloc)

# Examine WAP 248
WAP248 <- data_full %>% select(NOWAPS, source) %>% 
  filter(data_full$HighWAP=="WAP248")
plot(LATITUDE ~ LONGITUDE, data = data_full, pch = 20, col = "grey")
points(LATITUDE ~ LONGITUDE, data=WAP248[WAP248$source=="df_datatrain",], pch=20, col="blue")
points(LATITUDE ~ LONGITUDE, data=WAP248[WAP248$source=="df_datavalid",], pch=20, col="red")

# Let's remove it 
data_full<-data_full %>% select(-WAP248)     # 325 to 324 columns



#### D. EXPLORATION PER PHONE ----------------------------------------------------------




#### E. EXPLORATION PER USER ----------------------------------------------------------
#### F. EXPLORATION PER BUILDING & FLOOR----------------------------------------------------------





#### Z. OTHERS  ----------------------------------------------------------
#Let's compare locations
df_datatrain<-df_datatrain %>% mutate(Position=group_indices(df_datatrain, LATITUDE, LONGITUDE, FLOOR))
df_datavalid<-df_datavalid %>% mutate(Position=group_indices(df_datavalid, LATITUDE, LONGITUDE, FLOOR))

max(df_datatrain$Position)    # <- We have different positions in Train and Valid!!  O.O!!
max(df_datavalid$Position)
max(data_full$Position)

##### 3.2. Add a variable combining Build + Floor ##### -------------------------
# Add variable Build_Floor
Data_Full$Build_floorID<-as.factor(group_indices(Data_Full, BUILDINGID, FLOOR))

unique(Data_Full$Build_floorID) # <- 13 different floors 

check_ID<-Data_Full%>%
  arrange(BUILDINGID, FLOOR)%>%
  distinct(BUILDINGID, FLOOR, Build_floorID)%>%
  select(BUILDINGID, FLOOR, Build_floorID) #sIDs assigned sequentially. OK!!

rm(check_ID)

#### 4. EXPLORING STRANGE "THINGS"####
# Are there same HighWAP in different Buildings?
WAPS_Recoloc<-Data_Full %>%
  select(HighWAP, BUILDINGID, source) %>%
  distinct(HighWAP, BUILDINGID, source)

RepWAPS<-WAPS_Recoloc %>% distinct(HighWAP, BUILDINGID)
RepWAPS<-sort(RepWAPS$HighWAP[duplicated(RepWAPS$HighWAP)]) 
RepWAPS       # WAP 248 is highwap in the 3 buildings!!

# Examine WAP 248
WAP248<-Data_Full[Data_Full$HighWAP=="WAP248",313:326]
plot(LATITUDE ~ LONGITUDE, data = Data_Full, pch = 20, col = "grey")
points(LATITUDE ~ LONGITUDE, data=WAP248[WAP248$source=="DataTrain",], pch=20, col="blue")
points(LATITUDE ~ LONGITUDE, data=WAP248[WAP248$source=="DataValid",], pch=20, col="red")
