library(rjson)
library(stringr)
library(tm)
library(RTextTools)
source("match_annote_functions.R")
source("textmining_functions.R")

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
pathway_terms <- matching_keyterms(grantInfo, pathway,F)
pwgroup_terms <- matching_keyterms(grantInfo, pathway_group,F)
moleculartarget_terms <- matching_keyterms(grantInfo, molecular_target_group,F)
mtgroup_terms <- matching_keyterms(grantInfo, molecular_target,F)
metastage_terms <- matching_keyterms(grantInfo, metastasis_stage,F)
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

#compare human annotated vs machine
pw_compare <- compare("Pathway","match_pathway","new_jsons")
pwgroup_compare<-compare("Pathway (Group)","match_pwgroup","new_jsons")
mtgroup_compare<-compare("Molecular Target (Group)","match_mtgroups","new_jsons")
mt_compare<-compare("Molecular Target","match_mts","new_jsons")
metastage_compare<-compare("*Metastasis stage","match_metastage","new_jsons")

#Get accuracy of matching
getvalues(pw_compare)
getvalues(pwgroup_compare)
getvalues(mtgroup_compare)
getvalues(mt_compare)
getvalues(metastage_compare)

#---------------------------------------------------------
# Normalize data then do workflow
#---------------------------------------------------------
mainData <- as.vector(paste(grantInfo$AwardTitle,grantInfo$TechAbstract))
mainData <- toAmericanEnglish(mainData)
norm_mainData <- normalise_text(mainData)

#Vectorize annotation categories
Pathway <- as.vector(grantInfo$Pathway)
mt <- as.vector(grantInfo$Molecular.Target)
pwgroup <- as.vector(grantInfo$Pathway..Group.)
mtgroup <- as.vector(grantInfo$Molecular.Target..Group.)
meta <- as.vector(grantInfo$X.Metastasis.stage)

#Normalize human annotated categories
norm_Pathway <- normalise_text(Pathway)
norm_pwgroup <- normalise_text(pwgroup)
norm_mtgroup <- normalise_text(mtgroup)
norm_mt <- normalise_text(mt)
norm_meta <- normalise_text(meta)

#Process normalised annotation lists
list_norm_pw <- trim(unlist(unique(norm_Pathway),use.names = F))
list_norm_pw<- list_norm_pw[list_norm_pw != ""]

list_norm_pwgroup <- trim(unlist(unique(norm_pwgroup),use.names = F))
list_norm_pwgroup <- list_norm_pwgroup[list_norm_pwgroup!=""]

list_norm_mt <- trim(unlist(unique(norm_mt),use.names = F))
list_norm_mt <- list_norm_mt[list_norm_mt!=""]

list_norm_mtgroup <- trim(unlist(unique(norm_mtgroup),use.names = F))
list_norm_mtgroup <- list_norm_mtgroup[list_norm_mtgroup!=""]

list_norm_meta <- trim(unlist(unique(norm_meta),use.names = F))
list_norm_meta <- list_norm_meta[list_norm_meta!=""]

#Match keyterms
norm_pw_match <- matching_keyterms(norm_mainData,list_norm_pw,T)
norm_pwgroup_match <- matching_keyterms(norm_mainData,list_norm_pwgroup,T)
norm_mt_match <- matching_keyterms(norm_mainData,list_norm_mt,T)
norm_mtgroup_match <- matching_keyterms(norm_mainData, list_norm_mtgroup,T)
norm_meta_match <- matching_keyterms(norm_mainData, list_norm_meta,T)



### Normalized add to JSON
addition<-sapply(c(1:2237), function(x) {
  temp <- fromJSON(file=sprintf("dataexample/json/%d.json",x))
  temp$match_pathway = paste0(norm_pw_match[[x]],collapse = ",")
  temp$match_pwgroup = paste0(norm_pwgroup_match[[x]],collapse = ",")
  temp$match_mts = paste0(norm_mt_match[[x]],collapse = ",")
  temp$match_mtgroups = paste0(norm_mtgroup_match[[x]],collapse = ",")
  temp$match_metastage = paste0(norm_meta_match[[x]],collapse = ",")
  sink(sprintf("dataexample/Norm_json/%d.json",x))
  cat(toJSON(temp))
  sink()
})


#COMPARE
pwn_compare <- compare("Pathway","match_pathway","Norm_json",T)
pwgroupn_compare<-compare("Pathway (Group)","match_pwgroup","Norm_json",T)
mtgroupn_compare<-compare("Molecular Target (Group)","match_mtgroups","Norm_json",T)
mtn_compare<-compare("Molecular Target","match_mts","Norm_json",T)
metastagen_compare<-compare("*Metastasis stage","match_metastage","Norm_json",T)

getvalues(pwn_compare)
getvalues(pwgroupn_compare)
getvalues(mtn_compare)
getvalues(mtgroupn_compare)
getvalues(metastagen_compare)
