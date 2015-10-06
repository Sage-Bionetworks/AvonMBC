library(tm)
install.packages("RTextTools")
library(RTextTools)
grantInfo$Pathway
grantInfo<-read.csv("metastatic_grant_binary.csv",stringsAsFactors=F)
#Normalize Data
doc_matrix <- create_matrix(grantInfo$TechAbstract, language="english",
                            stemWords=TRUE, removeSparseTerms=.998)

#Create container 
container <- create_container(doc_matrix, grantInfo$pw_binary, trainSize=1:1700,
                              testSize=1701:2237, virgin=FALSE)

#Train data
SVM <- train_model(container,"SVM")
GLMNET <- train_model(container,"GLMNET")
MAXENT <- train_model(container,"MAXENT")
SLDA <- train_model(container,"SLDA")
BOOSTING <- train_model(container,"BOOSTING")
BAGGING <- train_model(container,"BAGGING")
RF <- train_model(container,"RF")
NNET <- train_model(container,"NNET")
TREE <- train_model(container,"TREE")

#analyze
analytics <- create_analytics(container, SVM_CLASSIFY)
GLMNET_CLASSIFY <- classify_model(container, GLMNET)
MAXENT_CLASSIFY <- classify_model(container, MAXENT)
SLDA_CLASSIFY <- classify_model(container, SLDA)
BOOSTING_CLASSIFY <- classify_model(container, BOOSTING)
BAGGING_CLASSIFY <- classify_model(container, BAGGING)
RF_CLASSIFY <- classify_model(container, RF)
NNET_CLASSIFY <- classify_model(container, NNET)
TREE_CLASSIFY <- classify_model(container, TREE)

#create analysis
analytics <- create_analytics(container,cbind(SVM_CLASSIFY, SLDA_CLASSIFY,BOOSTING_CLASSIFY, BAGGING_CLASSIFY,RF_CLASSIFY, GLMNET_CLASSIFY,NNET_CLASSIFY, TREE_CLASSIFY,MAXENT_CLASSIFY))

#Give summary
topic_summary <- analytics@label_summary
alg_summary <- analytics@algorithm_summary
ens_summary <-analytics@ensemble_summary
doc_summary <- analytics@document_summary


