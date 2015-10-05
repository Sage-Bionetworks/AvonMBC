library(tm)
install.packages("RTextTools")
library(RTextTools)
grantInfo$Pathway
grantInfo<-read.csv("Metastatic_grant.csv",stringsAsFactors=F)
#Normalize Data
doc_matrix <- create_matrix(grantInfo$TechAbstract, language="english", removeNumbers=TRUE,
                            stemWords=TRUE, removeSparseTerms=.998)
#Remove NA
grantInfo$Pathway[which(is.na(grantInfo$Pathway))]<-"Not specified"

#Create container 
container <- create_container(doc_matrix, grantInfo$Pathway, trainSize=1:1700,
                              testSize=1701:2237, virgin=FALSE)

#Train data
SVM <- train_model(container,"SVM")

