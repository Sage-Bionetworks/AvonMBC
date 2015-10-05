library(rjson)
library(stringr)
#--------------------------------------------------------
#get Specific annotation
#--------------------------------------------------------
trim <- function (x) gsub("^\\s+|\\s+$", "", x)

getAnnote <- function(filenames,type) {
  annotes<-lapply(filenames, function(x) {
    temp <- fromJSON(file=x)
    return(temp[sprintf("%s",type)])
  })
  return(unlist(annotes,use.names = F))
}

#------------------------------------
#Extract all unique pathways
#------------------------------------
annote_extract <- function(annote) {
  #Sub parenthesis with ","
  parenthesis <- gsub("\\(|\\)",",",annote)
  parenthesis <- parenthesis[!duplicated(tolower(parenthesis))]
  parenthesis <- unlist(strsplit(parenthesis,","))
  
  #Remove white spaces
  newlist<-trim(parenthesis)
  newlist <- newlist[!duplicated(tolower(newlist))]
  
  #Replace "or","and", and "/" with ","
  process <- gsub(" or ", ",", newlist,fixed=T)
  process <- gsub(" and ", ",",process, fixed=T)
  process <- gsub("/",",",process, fixed=T)
  
  #Split strings on ","
  process <- unlist(strsplit(process,","))
  process<- process[!duplicated(tolower(process))]
  
  #Manually curate
  final<-process[process != "-"]
  final <- final[final != "a"]
  final <- final[final != "n"]
  final <- final[final != "K"]
  final <- final[final != "3"]
  final <- final[final != "-1"]
  final <- final[final != "-2"]
  final <- final[final != ""]
  return(final)
}

#---------------------------------------
#Match top 5 key terms
#--------------------------------------
matching_keyterms <- function(grantInfo,list_annotes) {
  keyterms<- sapply(paste(grantInfo$TechAbstract,grantInfo$AwardTitle,sep=","), function(x) {
    words <- sapply(list_annotes, function(y) {
      #coll searches for just the word
      return(str_count(x,sprintf("\\b%s\\b",y)))
    })
    return(names(sort(words[words>0],decreasing = T)[1:5]))
  })
  return(keyterms)
}

#--------------------------------------
#Compare keyterms with human annotated
#--------------------------------------
compare <- function(human_annote, machine_annote) {
  accuracy<-lapply(c(1:length(new_files)), function(x) {
    temp <- fromJSON(file=sprintf("dataexample/new_jsons/%d.json",x))
    machine<-unlist(temp[sprintf("%s",machine_annote)],use.names=F)
    human <- unlist(temp[sprintf("%s",human_annote)],use.names=F)
    toms<- unlist(strsplit(machine,","))
    num<-sapply(toms, function(y) {
      grep(sprintf("\\b%s\\b",y),human,ignore.case =T)
    })
    if (is.null(human)) {
      return(0.5)
    } else if (human == "-" || tolower(human) =="not specified" || tolower(human) == "n/a") {
      return(0.5)
    } else {
      return(sum(unlist(num)))
    }
  })
  return(accuracy)
}
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

