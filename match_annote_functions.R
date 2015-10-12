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
matching_keyterms <- function(grantInfo,list_annotes,normalized) {
  if (!normalized) { 
    grants <- paste(grantInfo$TechAbstract,grantInfo$AwardTitle,sep=",")
  } else {
    grants <- grantInfo
  }
  keyterms<- apply(as.matrix(grants),1, function(x) {
    words <- sapply(list_annotes, function(y) {
      #coll searches for just the word
      return(str_count(x,sprintf("\\b%s\\b",y)))
    })
    if (normalized) {
      return(names(words[words>0]))
    } else {
      return(names(sort(words[words>0],decreasing = T)[1:5]))
    }
  })
  return(keyterms)
}

#--------------------------------------
#Compare keyterms with human annotated
#--------------------------------------
compare <- function(human_annote, machine_annote,folder,noNA) {
  accuracy<-lapply(c(1:2237), function(x) {
    temp <- fromJSON(file=sprintf("dataexample/%s/%d.json",folder,x))
    machine<-unlist(temp[sprintf("%s",machine_annote)],use.names=F)
    human <- unlist(temp[sprintf("%s",human_annote)],use.names=F)
    toms<- unlist(strsplit(machine,","))
    num<-sapply(toms, function(y) {
      grep(sprintf("\\b%s\\b",y),human,ignore.case =T)
    })
    lower <- tolower(human)
    if (noNA) {
      if (is.null(human)) {
        return(0.5)
      } else if (human == "-" || lower =="not specified" || lower == "n/a"  ||
                 lower == "other/not specified" || lower == "not relevant" ||
                 lower == "no abstract" || lower == "n/a (chemo delivery system)" || 
                 lower == "n/a (imaging of mets)"|| lower == "n/a (imaging)" || 
                 lower =="n/a (imaging of mets) & imaging herceptin delivery"||
                 lower == "n/a (therapy delivery system)" || lower == "n/a surgery"|| 
                 lower == "none"|| lower =="not applicable" || 
                 lower == "0_no specific target") {
        return(0.5)
      } #else {
        #return(sum(unlist(num)))
      #}
    } 
    return(sum(unlist(num)))
  })
  return(accuracy)
}


#------------------------------
# Getting Accuracy of matching
#------------------------------

getvalues <- function(compare) {
  print(paste("accuracy with NA:",mean(compare>0.5),sep=" "))
  print(paste("# NA:",sum(compare==0.5),sep=" "))
  temp<-compare[compare!=0.5]
  print(paste("accuracy without NA:",mean(temp>0),sep=" "))
}
