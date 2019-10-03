# WIFI Positioning
Many real-world applications need to know the localization of a user in the world to provide their services. Outdoor localization problem can be solved very accurately thanks to the inclusion of GPS sensors into the mobile devices. However, indoor localization is still an open problem due mainly to the loss of GPS signal in indoor environments. For this reason, we evaluated the application of machine learning techniques to this problem, replacing the GPS signal with the WAPS signal.

![UJI](https://github.com/Nell87/Wifi_Positioning/blob/master/report/UJI.png?raw=true)

For this purpose, we're going to use the UJIIndoorLoc database. It covers three buildings of Universitat Jaume I with four or more floors. You can read more about this dataset [here](https://archive.ics.uci.edu/ml/datasets/ujiindoorloc).

# TECHNICAL APPROACH
This is only a summary of the process, If you want to know more details, you can read the report in [saramarlop.com](http://saramarlop.com/)

## 1. FIRST STEPS

- **Exploratory analysis**
- **Cleaning and preparing datasets**:
  - Remove duplicated observations
  - Change the value of WAPS = 100 to -110
  - Review the type of variables
  - Get rid of the user 6
  - Remove “near-zero variance” predictors and registers
  - Remove WAP 248
- **Run the CRUD_app_shiny.R code**. The credentials.R and the CRUD_app_shiny.R files must be in the same folder. 

## 2. FEATURE ENGINEERING
It’s about creating new input features from the existing ones (we will use them in our predictions):
- The highes WAP
- The higher RSSI
- A new ID combining building + Floor

## 3. MODELING & ERROR ANALYSIS 
- We run three different models: rf, knn and svm
- We analyzed the errors based on the confusion matrix (classification) and the distribution on the errors (regression)
