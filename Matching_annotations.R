library(rjson)
library(stringr)
source("match_annote_functions.R")
#--------------------------------------------------------
#    Workflow
#--------------------------------------------------------
#Get list of files and remove first index, because it is 0.json
f<-list.files("dataexample/json",full.names = T)
f[-1]
#----------------------
#Get human annotations
#----------------------
pathway <- getAnnote(f,"Pathway")
pathway_group <- getAnnote(f,"Pathway (Group)")
molecular_target_group <- getAnnote(f,"Molecular Target (Group)")
molecular_target <- getAnnote(f,"Molecular Target")
metastasis_stage <- getAnnote(f,"*Metastasis stage")
#metastasis <- getAnnote(f,"Metastasis Y/N")

#--------------------------
#Extract unique annotations
#--------------------------
pathway <- annote_extract(pathway)
pathway_group <- annote_extract(pathway_group)
molecular_target_group <- annote_extract(molecular_target_group)
molecular_target <- annote_extract(molecular_target)
metastasis_stage <- annote_extract(metastasis_stage)
#metastasis <- annote_extract(f,"Metastasis Y/N")
write.table(pathway,"pathway_dict.csv",quote=F,row.names=F,col.names = F)
write.table(pathway_group,"pwgroup_dict.csv",quote=F,row.names=F,col.names = F)
write.table(molecular_target,"mt_dict.csv",quote=F,row.names=F,col.names = F)
write.table(molecular_target_group,"mtgroup_dict.csv",quote=F,row.names=F,col.names = F)
write.table(metastasis_stage,"metastage_dict.csv",quote=F,row.names=F,col.names = F)

#Get grants
grantInfo <- read.csv("Metastatic_grant.csv")

#----------------------
#Matching keyterms
#----------------------
pathway_terms <- matching_keyterms(grantInfo, pathway)
pwgroup_terms <- matching_keyterms(grantInfo, pathway_group)
moleculartarget_terms <- matching_keyterms(grantInfo, molecular_target_group)
mtgroup_terms <- matching_keyterms(grantInfo, molecular_target)
metastage_terms <- matching_keyterms(grantInfo, metastasis_stage)
#meta_terms <- matching_keyterms(grantInfo, list_annotes)

#--------------------------------------
#Add keyterms to json
#-----------------------------------

addition<-sapply(c(1:length(f)), function(x) {
  temp <- fromJSON(file=sprintf("dataexample/json/%d.json",x))
  temp$match_pathway = paste0(pathway_terms[,x],collapse = ",")
  temp$match_pwgroup = paste0(pwgroup_terms[,x],collapse = ",")
  temp$match_mts = paste0(moleculartarget_terms[,x],collapse = ",")
  temp$match_mtgroups = paste0(mtgroup_terms[,x],collapse = ",")
  temp$match_metastage = paste0(metastage_terms[,x],collapse = ",")
  sink(sprintf("dataexample/new_jsons/%d.json",x))
  cat(toJSON(temp))
  sink()
})

pw_compare <- compare("Pathway","match_pathway")
pwgroup_compare<-compare("Pathway (Group)","match_pwgroup")
mtgroup_compare<-compare("Molecular Target (Group)","match_mtgroups")
mt_compare<-compare("Molecular Target","match_mts")
metastage_compare<-compare("*Metastasis stage","match_metastage")

getvalues(pw_compare)
getvalues(pwgroup_compare)
getvalues(mtgroup_compare)
getvalues(mt_compare)
getvalues(metastage_compare)

getvalues <- function(compare) {
  print(paste("accuracy with NA:",mean(compare>0.5),sep=" "))
  print(paste("# NA:",sum(compare==0.5),sep=" "))
  temp<-compare[compare!=0.5]
  print(paste("accuracy without NA:",mean(temp>0),sep=" "))
}

#---------------------------------------------------------
# Normalize data then do workflow
#---------------------------------------------------------
library(tm)
library(RTextTools)
source("textmining_functions.R")

mainData <- as.vector(paste(grantInfo$AwardTitle,grantInfo$TechAbstract))
mainData <- toAmericanEnglish(mainData)
norm_mainData <- normalise_text(mainData)

Pathway <- as.vector(grantInfo$Pathway)
norm_Pathway <- normalise_text(Pathway)
View(norm_mainData)
View(norm_Pathway)
class(norm_Pathway)
list_norm_pw <- unlist(unique(norm_Pathway),use.names = F)


