library(rjson)
library(stringr)
library(tm)
library(RTextTools)
source("R/match_annote_functions.R")
source("R/textmining_functions.R")
# --------------------------------------------------------
# --------------------------------------------------------
#    Workflow
# --------------------------------------------------------
# --------------------------------------------------------



# ------------------------------------------------------------------
# Extract MBC provided annotations
# ------------------------------------------------------------------

#Get list of files and remove first index, because it is 0.json
f<-list.files("dataexample/json",full.names = T)

pathway <- getAnnote(f,"Pathway")
molecular_target <- getAnnote(f,"Molecular Target")

#pathway_group <- getAnnote(f,"Pathway (Group)")
#molecular_target_group <- getAnnote(f,"Molecular Target (Group)")
#metastasis_stage <- getAnnote(f,"*Metastasis stage")
#metastasis <- getAnnote(f,"Metastasis Y/N")

# ------------------------------------------------------------------------------
#Extract unique annotations
# ------------------------------------------------------------------------------
pathway <- annote_extract(pathway)
molecular_target <- annote_extract(molecular_target)

#pathway_group <- annote_extract(pathway_group)
#molecular_target_group <- annote_extract(molecular_target_group)
#metastasis_stage <- annote_extract(metastasis_stage)
#metastasis <- annote_extract(f,"Metastasis Y/N")

#Write out MBC annotation dictionaries
write.table(pathway,"pathway_dict.csv",quote=F,row.names=F,col.names = F)
write.table(molecular_target,"mt_dict.csv",quote=F,row.names=F,col.names = F)

#write.table(pathway_group,"pwgroup_dict.csv",quote=F,row.names=F,col.names = F)
#write.table(molecular_target_group,"mtgroup_dict.csv",quote=F,row.names=F,col.names = F)
#write.table(metastasis_stage,"metastage_dict.csv",quote=F,row.names=F,col.names = F)




# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Matching keyterms - Not normalized
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#Get grants
grantInfo <- read.csv("Metastatic_grant.csv")
pathway_terms <- matching_keyterms(grantInfo, pathway,F)
pwgroup_terms <- matching_keyterms(grantInfo, pathway_group,F)
moleculartarget_terms <- matching_keyterms(grantInfo, molecular_target_group,F)
mtgroup_terms <- matching_keyterms(grantInfo, molecular_target,F)
metastage_terms <- matching_keyterms(grantInfo, metastasis_stage,F)
#meta_terms <- matching_keyterms(grantInfo, list_annotes)

# -------------------------------------------------------------
# Add keyterms to json
# -------------------------------------------------------------

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


# ---------------------------------------------------------
# ---------------------------------------------------------
# Normalize data then do workflow
# ---------------------------------------------------------
# ---------------------------------------------------------

grantInfo <- read.csv("Metastatic_grant.csv",stringsAsFactors = F)
mainData <- as.vector(paste(grantInfo$AwardTitle,grantInfo$TechAbstract))
norm_mainData <- normalise_text(mainData)
norm_mainData[1]

pw_pwgroup_mapped <- read.csv("dictionary/pw_pwgroup_mapping.csv",stringsAsFactors = F)

#Normalize human annotated categories
norm_pw <- normalise_text(pw_pwgroup_mapped$pw)

norm_pw
#Matching keyterms has normalize parameter (default = True), Input original grantInfo 
#along with pw_pwgroup mapped
norm_pw_match<- matching_keyterms(grantInfo,pw_pwgroup_mapped[,1],removeNumbers = T)

#Processed normalised annotation lists
list_norm_pw<- read.csv('dictionary/pw_dict_normed.csv')
#Will match the pathway group to the matched pathway, no need to match pathway group to text
#Build on the dictionary.. Correlation between molecular target and pathway... then molecular target
#group can be matched with molecular target

list_norm_mt <- trim(unlist(unique(norm_mt),use.names = F))
list_norm_mt <- list_norm_mt[list_norm_mt!=""]

list_norm_meta <- trim(unlist(unique(norm_meta),use.names = F))
list_norm_meta <- list_norm_meta[list_norm_meta!=""]


#Match keyterms
norm_pw_match <- matching_keyterms(grantInfo,pw_pwgroup_mapped[,1],removeNumbers = F)
#Pw group and molecular target group can map to pathway/molecular target
norm_pwgroup_match <- matching_keyterms(grantInfo,list_norm_pwgroup,T)
norm_mt_match <- matching_keyterms(grantInfo,list_norm_mt,T)
norm_mtgroup_match <- matching_keyterms(grantInfo, list_norm_mtgroup,T)
norm_meta_match <- matching_keyterms(grantInfo, list_norm_meta,T)


### Normalized add to JSON
addition<-sapply(c(1:2237), function(x) {
  temp <- fromJSON(file=sprintf("dataexample/json/%d.json",x))
  temp$match_pathway = paste0(norm_pw_match[[x]],collapse = ", ")
  #temp$match_pwgroup = paste0(norm_pwgroup_match[[x]],collapse = ",")
  #temp$match_mts = paste0(norm_mt_match[[x]],collapse = ",")
  #temp$match_mtgroups = paste0(norm_smtgroup_match[[x]],collapse = ",")
  #temp$match_metastage = paste0(norm_meta_match[[x]],collapse = ",")
  sink(sprintf("dataexample/Norm_json/%dpw.json",x))
  cat(toJSON(temp))
  sink()
})


#COMPARE
pwn_compare <- compare("Pathway","match_pathway","Norm_json",T,removeNumbers = T)
pwgroupn_compare<-compare("Pathway (Group)","match_pwgroup","Norm_json",T)
mtgroupn_compare<-compare("Molecular Target (Group)","match_mtgroups","Norm_json",T)
mtn_compare<-compare("Molecular Target","match_mts","Norm_json",T)
metastagen_compare<-compare("*Metastasis stage","match_metastage","Norm_json",T)

getvalues(pwn_compare)
getvalues(pwgroupn_compare)
getvalues(mtn_compare)
getvalues(mtgroupn_compare)
getvalues(metastagen_compare)


# --------------------------------------------------------------------------
# Matching to BioCarta, KEGG, and WikiPathways
# --------------------------------------------------------------------------
pathways = read.csv("pathways/new_all_pathways.txt",sep="+",header=F,stringsAsFactors = F)
grantInfo = read.csv("metastatic_grant_binary.csv")
mainData <- as.vector(paste(grantInfo$AwardTitle,grantInfo$TechAbstract))
normalized_main <- normalise_text(mainData,removenumbers = F,stemDoc = F)

f<-sapply(c(1:length(normalized_main$abstracts)), function(i) {
  x <- normalized_main$abstracts[i]
  tokens <- unlist(strsplit(x," "))
  t<-sapply(pathways$V2, function(y) {
    genes <- tolower(unlist(strsplit(y,",")))
    #returns a list of sum of times genes appear in abstract, and the gene that appears
    return(list(sum = sum(genes%in%tokens),gene = genes[genes%in%tokens]))
  })
  sums <- unlist(t[1,])
  maxValue <- max(names(table(sums)))
  #Pathway that goes along with the gene matched
  match <- pathways[which(t[1,]==maxValue),1]
  intersect_genes <- t[2,which(t[1,]==maxValue)]
  pathways = paste(match,collapse=", ")
  matched_gene = paste(toupper(unique(intersect_genes)),collapse=", ")
  if (maxValue==0) { #If no match, specify
    pathways = "No genes mentioned"
    matched_gene = "No genes mentioned"
  }
  index <- which(x==normalized_main$abstracts)
  temp <- fromJSON(file=sprintf("dataexample/new_jsons/%d.json",index[1]))
  temp$dict_pathways <- pathways
  temp$dict_genes <- matched_gene
  
  sink(sprintf("dataexample/match_dict_jsons/%d.json",i))
  cat(toJSON(temp))
  sink()
})

##Accuracy of match
all <- list.files("dataexample/match_dict_jsons",full.names=T)

compare <- sapply(all, function(x) {
  temp <- fromJSON(file=x)
  if (temp$dict_pathways == "No genes mentioned") {
    return(temp$Pathway)
  } else {
    return("+")
  }
})

check <- compare[compare != "+"]
sum(check=="-")
sum(check=="n/a")
sum(check=="N/A")
sum(compare!="+")

sort(unlist(unique(check)))

View(check)
unlist(check)
sum(compare == "+")





#---------------------------------------------------------
# Grant normalized using python
#---------------------------------------------------------
normed <- read.csv("no")

