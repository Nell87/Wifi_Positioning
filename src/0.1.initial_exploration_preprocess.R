
#### 0. INCLUDES  --------------------------------------------------------------------

#Load Libraries: p_load can install, load,  and update packages
if(require("pacman")=="FALSE"){
  install.packages("pacman")
} 

pacman::p_load(rstudioapi,dplyr, lubridate, caret, reshape2)

# Setwd (1º current wd where is the script, then we move back to the 
# general folder)
current_path = getActiveDocumentContext()$path 
setwd(dirname(current_path))
setwd("..")
rm(current_path)

# Load Data
df_datatrain <- read.csv2("data/trainingData.csv", header=TRUE, sep=",",  stringsAsFactors=FALSE, na.strings=c("NA", "-", "?"))
df_datavalid<-read.csv2("data/validationData.csv", header=TRUE, sep=",",  stringsAsFactors=FALSE, na.strings=c("NA", "-", "?"))

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


#### C. EXPLORATORY ANALYSIS  ----------------------------------------------------------
##### C.1. RSSI ######   

# Change value of WAPS= 100 to WAPS=-110 __________________________
data_full[,WAPS] <- sapply(data_full[,WAPS],function(x) ifelse(x==100,-110,x))

# Let's see if we have signal >=30 __________________________
# Let's reshape 
data_full_melt<-melt(data_full, id.vars=c("id", "LATITUDE", "LONGITUDE", "FLOOR", "BUILDINGID", 
                                        "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID",
                                        "TIMESTAMP", "source"), variable.name = "WAP", value.name="RSSI")

# Let's see if we have signal >=30
df_datatrain_RSSI<-data_full_melt %>%
  filter(source=="df_datatrain" & RSSI>=-30) %>%
  select(RSSI, USERID, PHONEID) %>%
  group_by(RSSI, USERID, PHONEID) %>%
  summarise(N=n())                     # <- # We have values higher than -30 in Train (not in Test). 
                                       # It doesn't make sense! x_X 

# Let's plot these signals
df_datatrain_RSSI<-data_full_melt %>%
  filter(source=="df_datatrain" & RSSI>=-30) %>%
  select(RSSI, USERID, PHONEID, LATITUDE, LONGITUDE, BUILDINGID, FLOOR) %>%
  group_by(RSSI, USERID, PHONEID,LATITUDE, LONGITUDE, BUILDINGID, FLOOR) %>%
  summarise(N=n())  

ggplot(df_datatrain_RSSI, aes(x = LONGITUDE, y = LATITUDE))  + 
  geom_point(aes(color=USERID)) + 
  facet_wrap(~FLOOR)      # <- USERID 6 IS WRONG!!!!  Let's see the percentage of errors

# Let's explore USERID 6: how many records have a signal >-30?
df_datatrain_user6<-data_full %>%
  filter(source=="df_datatrain" & USERID==6) 

df_datatrain_user6["max_signal"] <- apply(df_datatrain_user6[WAPS], 1, max)

df_datatrain_user6$state_signal<- ifelse(df_datatrain_user6$max_signal>=-30, "Strange", "Ok")

ggplot(df_datatrain_user6, aes(x=LONGITUDE, y=LATITUDE, color=state_signal)) + 
  geom_point() # There are many weird signals

df_datatrain_user6 %>% 
  select(state_signal) %>%
  group_by(state_signal) %>%
  summarise(N=n()/nrow(df_datatrain_user6)) # 44% is weird, so let's remove user 6

rm(df_datatrain_RSSI, df_datatrain_user6, data_full_melt)
data_full <- data_full %>% filter(!USERID==6)
df_datatrain <- df_datatrain %>% filter(!USERID==6)

# Remove WAPS with no variance ____________________________________
# New WAPS
WAPS<-grep("WAP", names(data_full), value=T)

# Filter Rows with all RSSI = -110        <- From 19434 to 19361 row
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
# Let's compare locations
df_datatrain<-df_datatrain %>% mutate(Position=group_indices(df_datatrain, LATITUDE, LONGITUDE, FLOOR))
df_datavalid<-df_datavalid %>% mutate(Position=group_indices(df_datavalid, LATITUDE, LONGITUDE, FLOOR))

max(df_datatrain$Position)    # <- We have different positions in Train and Valid!!  O.O!!
max(df_datavalid$Position)

#Add a variable combining Build + Floor _________________________
# Add variable Build_Floor
data_full$Build_floorID<-as.factor(group_indices(data_full, BUILDINGID, FLOOR))

unique(data_full$Build_floorID) # <- 13 different floors 

check_ID<-data_full%>%
  arrange(BUILDINGID, FLOOR)%>%
  distinct(BUILDINGID, FLOOR, Build_floorID)%>%
  select(BUILDINGID, FLOOR, Build_floorID) #sIDs assigned sequentially. OK!!

rm(check_ID)


#### Z. OTHERS  ----------------------------------------------------------
# New datasets prepared for modeling 
Data_FullSplit<-split(data_full, data_full$source)
list2env(Data_FullSplit, envir=.GlobalEnv)
rm(Data_FullSplit)

write.csv(df_datatrain, "trainingData_prepared.csv")
write.csv(df_datatrain, "validationData_prepared.csv")
