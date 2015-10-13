library(xlsx)
mt_excel<-read.xlsx("MetastaticBC_Grants_CodingFile_Complete_Sagebase.xlsx",4)
pw_excel<-read.xlsx("MetastaticBC_Grants_CodingFile_Complete_Sagebase.xlsx",5)
grantInfo <- read.csv("Metastatic_grant.csv",check.names = F)
pathways<- colnames(grantInfo)[grep("^KW",colnames(grantInfo))]
pathways <- substring(pathways,4)
pathways <- pathways[-1]
pathways
View(mt_excel$Molecular.Target)
mt_excel$Molecular.Target..Group.

write.csv(pathways,"dictionary/KW.csv",row.names = F)
read.csv("dictionary/pathway_dict.csv")

pw_checked<- read.csv("dictionary/pw_dict_check.csv")
write.csv(factor(pw_checked$x[!duplicated(tolower(pw_checked$x))]),"dictionary/pw_dict_check.csv",row.names = F)
