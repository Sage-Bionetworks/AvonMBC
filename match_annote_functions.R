
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
  keyterms<- sapply(grants, function(x) {
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