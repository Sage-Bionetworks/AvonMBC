library(tm)
library(RTextTools)

grantInfo<-read.csv("Metastatic_grant.csv",stringsAsFactors=F)
non_breastcancer <- read.csv("nonbreast_grants.csv",stringsAsFactors =F)

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# BREAST CANCER CLASSIFIER
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
grantInfo$bc <- 1
non_breastcancer$bc <- 0

allGrants <- rbind(grantInfo[,c("AwardTitle","TechAbstract","bc")],non_breastcancer[,c("AwardTitle","TechAbstract","bc")])

shuffled <- allGrants[sample(nrow(allGrants),nrow(allGrants)),]
#----------------------------------------------------------------------------------------------------------------------------------
# Match breast to new abstracts only 24% of these have "breast" in them and this is good, because these grants are not about
# breast cancer
#----------------------------------------------------------------------------------------------------------------------------------
tt = unname(sapply(non_breastcancer$TechAbstract, function(x) {
  if (grepl("breast",x)) {
    return(1)
  } else {
    return(0)
  }
}))
mean(tt)
nrow(shuffled)
#93% percent of these grants have "breast" in them
ff = unname(sapply(grantInfo$TechAbstract, function(y) {
  if (length(grep("breast",y))!=0)
    return(1)
  return(0)
}))
mean(ff)

#----------------------------------------------------------------------------------------------------------------------------------
#Normalize Data
doc_matrix <- create_matrix(paste(shuffled$AwardTitle,shuffled$TechAbstract), language="english",
                            stemWords=TRUE,removeNumbers = T, removeSparseTerms=.998)

#Create container 
container <- create_container(doc_matrix, grantInfo$pw_binary, trainSize=1:(nrow(shuffled)/2),
                              testSize= (nrow(shuffled)/2):nrow(shuffled), virgin=FALSE)


trainData <- function(group) {
  group = "meta_binary"
  binary <- unlist(grantInfo[sprintf("%s",group)],use.names=F)
  #Create container 
  container <- create_container(doc_matrix, binary, trainSize=1:1900,
                                testSize=1901:2237, virgin=FALSE)
  #Train data
  SVM <- train_model(container,"SVM")
  GLMNET <- train_model(container,"GLMNET")
  MAXENT <- train_model(container,"MAXENT")
  SLDA <- train_model(container,"SLDA")
  #BOOSTING <- train_model(container,"BOOSTING")
  #BAGGING <- train_model(container,"BAGGING")
  RF <- train_model(container,"RF")
  #NNET <- train_model(container,"NNET")
  TREE <- train_model(container,"TREE")
  
  #Create analysis
  SVM_CLASSIFY <- classify_model(container, SVM)
  GLMNET_CLASSIFY <- classify_model(container, GLMNET)
  MAXENT_CLASSIFY <- classify_model(container, MAXENT)
  SLDA_CLASSIFY <- classify_model(container, SLDA)
  #BOOSTING_CLASSIFY <- classify_model(container, BOOSTING)
  #BAGGING_CLASSIFY <- classify_model(container, BAGGING)
  RF_CLASSIFY <- classify_model(container, RF)
  #NNET_CLASSIFY <- classify_model(container, NNET)
  TREE_CLASSIFY <- classify_model(container, TREE)
  
  analytics <- create_analytics(container,cbind(SVM_CLASSIFY,
                                                #SLDA_CLASSIFY,
                                                GLMNET_CLASSIFY,
                                                RF_CLASSIFY, 
                                                #BOOSTING_CLASSIFY, 
                                                #BAGGING_CLASSIFY,
                                                #NNET_CLASSIFY, 
                                                #TREE_CLASSIFY,
                                                MAXENT_CLASSIFY))
  
  #Save analysis
  #topic_summary <- analytics@label_summary
  #alg_summary <- analytics@algorithm_summary
  #ens_summary <-analytics@ensemble_summarys
  doc_summary <- analytics@document_summary
  write.csv(doc_summary, sprintf("RTextTools_results/sparse_%s_DocumentSummary.csv",group))
  #write.csv(alg_summary, sprintf("RTextTools_results/sparse_%s_AlgSummary.csv",group))
  #write.csv(ens_summary, sprintf("RTextTools_results/sparse_%s_EnsSummary.csv",group))
  #write.csv(topic_summary, sprintf("RTextTools_results/sparse_%s_TopicSummary.csv",group))
}


#Accuracy function
get_accuracy <- function(document_summary) {
  f <- document_summary
  consensus <- mean(f$CONSENSUS_CODE == f$MANUAL_CODE)
  SVM <- mean(f$SVM_LABEL == f$MANUAL_CODE)
  GLM <- mean(f$GLMNET_LABEL == f$MANUAL_CODE)
  FOREST <- mean(f$FORESTS_LABEL == f$MANUAL_CODE)
  MAX <- mean(f$MAXENTROPY_LABEL == f$MANUAL_CODE)
  results <- data.frame(consensus,SVM, GLM, FOREST, MAX)
  return(results)
}

#-----------------------------------
#   TRAIN DATA
#-----------------------------------
trainData("pw_binary")
trainData("pwgroup_binary")
trainData("mtgroup_binary")
trainData("mt_binary")
trainData("meta_binary")

#----------------------------------------------------------------------
#   Get accuracy of RTextTools (sparse vs 0.998 sparse terms removed)
#----------------------------------------------------------------------
pwsparse_ds<-read.csv("RTextTools_results/sparse_pw_binary_DocumentSummary.csv")
get_accuracy(pwsparse_ds)
pw_ds <- read.csv("RTextTools_results/pw_binary_DocumentSummary.csv")
get_accuracy(pw_ds)

pwgroupsparse_ds <- read.csv("RTextTools_results/sparse_pwgroup_binary_DocumentSummary.csv")
get_accuracy(pwgroupsparse_ds)
pwgroup_ds <- read.csv("RTextTools_results/pwgroup_binary_DocumentSummary.csv")
get_accuracy(pwgroup_ds)

mtgroupsparse_ds <- read.csv("RTextTools_results/sparse_mtgroup_binary_DocumentSummary.csv")
get_accuracy(mtgroupsparse_ds)
mtgroup_ds <- read.csv("RTextTools_results/mtgroup_binary_DocumentSummary.csv")
get_accuracy(mtgroup_ds)

mtsparse_ds <- read.csv("RTextTools_results/sparse_mt_binary_DocumentSummary.csv")
get_accuracy(mtsparse_ds)
mt_ds <- read.csv("RTextTools_results/mt_binary_DocumentSummary.csv")
get_accuracy(mt_ds)

metasparse_ds <- read.csv("RTextTools_results/sparse_meta_binary_DocumentSummary.csv")
get_accuracy(metasparse_ds)
meta_ds <- read.csv("RTextTools_results/meta_binary_DocumentSummary.csv")
get_accuracy(meta_ds)





# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ANNOTATION CLASSIFIERS
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
grantInfo<-read.csv("metastatic_grant_binary.csv",stringsAsFactors=F)
non_breastcancer <- read.csv("nonbreast_grants.csv",stringsAsFactors =F)

#Normalize Data
doc_matrix <- create_matrix(grantInfo$TechAbstract, language="english",
                            stemWords=TRUE,removeNumbers = T, removeSparseTerms=.998)

#Create container 
container <- create_container(doc_matrix, grantInfo$pw_binary, trainSize=1:1700,
                              testSize=1701:2237, virgin=FALSE)


trainData <- function(group) {
  group = "meta_binary"
  binary <- unlist(grantInfo[sprintf("%s",group)],use.names=F)
  #Create container 
  container <- create_container(doc_matrix, binary, trainSize=1:1900,
                                testSize=1901:2237, virgin=FALSE)
  #Train data
  SVM <- train_model(container,"SVM")
  GLMNET <- train_model(container,"GLMNET")
  MAXENT <- train_model(container,"MAXENT")
  SLDA <- train_model(container,"SLDA")
  #BOOSTING <- train_model(container,"BOOSTING")
  #BAGGING <- train_model(container,"BAGGING")
  RF <- train_model(container,"RF")
  #NNET <- train_model(container,"NNET")
  TREE <- train_model(container,"TREE")
  
  #Create analysis
  SVM_CLASSIFY <- classify_model(container, SVM)
  GLMNET_CLASSIFY <- classify_model(container, GLMNET)
  MAXENT_CLASSIFY <- classify_model(container, MAXENT)
  SLDA_CLASSIFY <- classify_model(container, SLDA)
  #BOOSTING_CLASSIFY <- classify_model(container, BOOSTING)
  #BAGGING_CLASSIFY <- classify_model(container, BAGGING)
  RF_CLASSIFY <- classify_model(container, RF)
  #NNET_CLASSIFY <- classify_model(container, NNET)
  TREE_CLASSIFY <- classify_model(container, TREE)
  
  analytics <- create_analytics(container,cbind(SVM_CLASSIFY,
                                                #SLDA_CLASSIFY,
                                                GLMNET_CLASSIFY,
                                                RF_CLASSIFY, 
                                                #BOOSTING_CLASSIFY, 
                                                #BAGGING_CLASSIFY,
                                                #NNET_CLASSIFY, 
                                                #TREE_CLASSIFY,
                                                MAXENT_CLASSIFY))
  
  #Save analysis
  #topic_summary <- analytics@label_summary
  #alg_summary <- analytics@algorithm_summary
  #ens_summary <-analytics@ensemble_summarys
  doc_summary <- analytics@document_summary
  write.csv(doc_summary, sprintf("RTextTools_results/sparse_%s_DocumentSummary.csv",group))
  #write.csv(alg_summary, sprintf("RTextTools_results/sparse_%s_AlgSummary.csv",group))
  #write.csv(ens_summary, sprintf("RTextTools_results/sparse_%s_EnsSummary.csv",group))
  #write.csv(topic_summary, sprintf("RTextTools_results/sparse_%s_TopicSummary.csv",group))
}


#Accuracy function
get_accuracy <- function(document_summary) {
  f <- document_summary
  consensus <- mean(f$CONSENSUS_CODE == f$MANUAL_CODE)
  SVM <- mean(f$SVM_LABEL == f$MANUAL_CODE)
  GLM <- mean(f$GLMNET_LABEL == f$MANUAL_CODE)
  FOREST <- mean(f$FORESTS_LABEL == f$MANUAL_CODE)
  MAX <- mean(f$MAXENTROPY_LABEL == f$MANUAL_CODE)
  results <- data.frame(consensus,SVM, GLM, FOREST, MAX)
  return(results)
}

#-----------------------------------
#   TRAIN DATA
#-----------------------------------
trainData("pw_binary")
trainData("pwgroup_binary")
trainData("mtgroup_binary")
trainData("mt_binary")
trainData("meta_binary")

#----------------------------------------------------------------------
#   Get accuracy of RTextTools (sparse vs 0.998 sparse terms removed)
#----------------------------------------------------------------------
pwsparse_ds<-read.csv("RTextTools_results/sparse_pw_binary_DocumentSummary.csv")
get_accuracy(pwsparse_ds)
pw_ds <- read.csv("RTextTools_results/pw_binary_DocumentSummary.csv")
get_accuracy(pw_ds)

pwgroupsparse_ds <- read.csv("RTextTools_results/sparse_pwgroup_binary_DocumentSummary.csv")
get_accuracy(pwgroupsparse_ds)
pwgroup_ds <- read.csv("RTextTools_results/pwgroup_binary_DocumentSummary.csv")
get_accuracy(pwgroup_ds)

mtgroupsparse_ds <- read.csv("RTextTools_results/sparse_mtgroup_binary_DocumentSummary.csv")
get_accuracy(mtgroupsparse_ds)
mtgroup_ds <- read.csv("RTextTools_results/mtgroup_binary_DocumentSummary.csv")
get_accuracy(mtgroup_ds)

mtsparse_ds <- read.csv("RTextTools_results/sparse_mt_binary_DocumentSummary.csv")
get_accuracy(mtsparse_ds)
mt_ds <- read.csv("RTextTools_results/mt_binary_DocumentSummary.csv")
get_accuracy(mt_ds)

metasparse_ds <- read.csv("RTextTools_results/sparse_meta_binary_DocumentSummary.csv")
get_accuracy(metasparse_ds)
meta_ds <- read.csv("RTextTools_results/meta_binary_DocumentSummary.csv")
get_accuracy(meta_ds)