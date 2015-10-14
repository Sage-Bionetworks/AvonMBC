library(xlsx)
source("textmining_functions.R")
mt_excel<-read.xlsx("MetastaticBC_Grants_CodingFile_Complete_Sagebase.xlsx",4)
pw_excel<-read.xlsx("MetastaticBC_Grants_CodingFile_Complete_Sagebase.xlsx",5)
grantInfo <- read.csv("Metastatic_grant.csv",check.names = F)
#-----------------------------------
#extract keywords from excel
#-----------------------------------
pathways<- colnames(grantInfo)[grep("^KW",colnames(grantInfo))]
pathways <- substring(pathways,4)
pathways <- pathways[-1]
write.csv(pathways,"dictionary/KW.csv",row.names = F)

#-------------------------------------------------------------
# Molecular target - Molecular target group mapping
#-------------------------------------------------------------
mt_excel$Molecular.Target..Group.




#---------------------------------------------
# Mapping pathway to pathway group
#---------------------------------------------
#Hand curated pathway dictionary
t<-read.csv("dictionary/pathway_dict.csv")

pw_group_map <- apply(as.matrix(t$x),1, function(x) {
  groups <- pw_excel$Pathway..Group.[grep(sprintf("\\b%s\\b",x),pw_excel$Pathway)]
  return(paste(unique(groups),collapse=","))
})

pw_pwgroup_mapped<- cbind(as.character(t$x),pw_group_map)
#write.csv(pw_pwgroup_mapped,"dictionary/pw_pwgroup_mapping.csv",row.names=F)

#list_norm_pw<- normalise_text(list_norm_pw)

