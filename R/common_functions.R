library(tm)
library(rjson)
library(stringr)
library(RTextTools)
# ---------------------------------------------------------------------
# Text mining functions
# ---------------------------------------------------------------------
bioStopWords <- stopwords('english')
bioStopWords <- bioStopWords[bioStopWords!="no"]
bioStopWords <- bioStopWords[bioStopWords!='her']
bioStopWords <- bioStopWords[bioStopWords!="am"]
skipWords <- function(x) removeWords(x, c(bioStopWords))

#"abil","across","releas","accompani","chang","surprisingly",
#"accur","achiev","enrich","singl","constitut","almost","among","upon","attenu",
#"conjug","capabl","candid","day","alon","basi","account","event","annot",
#"thought","applic","appli","longer","infer","cohort","assess","certain",
#"convent","baselin","associ","area","caspas","affect","adjust","adjuv",
#"ultim","time","dual","cannon","copi","address","anim","aris","capac",
#"amount","common","administ","adjac","addit","play","indeed","distribut",
#"presenc","collectively","dure","calcul","childhood","along","additionally",
#"comprehens","discov","side","within","global","construct","depend","aberr",
#"accord","safeti","abund","administr","combin","angiogenesi","best","acquir",
#"activ","allow","specif"))

concatenate_and_split_hyphens <- function(x){gsub("\\s(\\w+)-(\\w+)\\s"," \\1\\2 \\1 \\2 ",x)}
removeAloneNumbers <- function(x){gsub("\\s\\d+\\s","", x)}
trim <- function (x) gsub("^\\s+|\\s+$", "", x)

strip.markup <- function(x){gsub("</?[A-Za-z]+>|<[A-Za-z]+\\s?/>"," ", x)}
strip.specialchar <- function(x){gsub("&#\\d+;"," ", iconv(x, "latin1", "UTF-8"))}
remove.punctuation <- function(x) {gsub("[[:punct:]]"," ",x )}

toAmericanEnglish <- function(x){
  gsub("tumour","tumor",x)
}
cleanFun <- function(htmlString) {
  return(gsub("<.*?>", "", htmlString))
}

normalise_text <- function(textVec,removenumbers=T,stemDoc=T) {
  #Must be a textVector that is passed in
  if (!is.vector(mainData)) 
    textVec <- as.vector(textVec)
  
  textVec <- toAmericanEnglish(tolower(strip.specialchar(strip.markup(textVec))))
  corpusraw <- Corpus(VectorSource(textVec))
  funs <- list(stripWhitespace,
               skipWords,
               remove.punctuation)
  if (removenumbers) {
    funs <- c(funs,removeNumbers)
  } 
  if (stemDoc) {
    funs <- c(funs,stemDocument)
  }
  funs <- c(funs,trim)
  #normalize
  corpus <- tm_map(corpusraw, FUN = tm_reduce, tmFuns = funs)
  
  ####Added this line to avoid error######
  corpus <- tm_map(corpus, PlainTextDocument)

  #Convert corpus to dataframe  
  dataframe<-data.frame(abstracts=unlist(sapply(corpus, `[`, "content")), 
                        stringsAsFactors=F)
  dataframe <- sapply(dataframe, trim)
  return(dataframe)
}


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Matching annotation functions
# ------------------------------------------------------------------
# ------------------------------------------------------------------

# --------------------------------------------------------
# get Specific annotation
# --------------------------------------------------------

getAnnote <- function(filenames,type) {
  annotes<-lapply(filenames, function(x) {
    temp <- fromJSON(file=x)
    return(temp[sprintf("%s",type)])
  })
  return(unlist(annotes,use.names = F))
}

# ------------------------------------
# Extract all unique pathways
# ------------------------------------
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

# ---------------------------------------
# Match top 5 key terms
# --------------------------------------
matching_keyterms <- function(text,annotes,normalized=T,removeNumbers=F) {
  keyterms<- apply(as.matrix(text),1, function(x) {
    words <- sapply(annotes, function(y) {
      #coll searches for just the word
      return(str_count(x,sprintf("\\b%s\\b",y)))
    })
    return(names(words[words>0]))
  })
  return(keyterms)
}

#--------------------------------------
#Compare keyterms with human annotated
#--------------------------------------
#pwn_compare <- compare("Pathway","match_pathway","Norm_json",noNA = T)

compare <- function(human_annote, machine_annote,folder,noNA=F) {
  accuracy<-lapply(c(1:2237), function(x) {
    temp <- fromJSON(file=sprintf("dataexample/%s/%dnew.json",folder,x))
    machine<-unlist(temp[sprintf("%s",machine_annote)],use.names=F)
    human <- unlist(temp[sprintf("%s",human_annote)],use.names=F)
    lower <- tolower(human)
    #Text still has to be a little bit normalized, just don't remove numbers/stem Doc
    if (folder == "Norm_json") {
      human <- normalise_text(human)
      human <- unlist(unique(human),use.names = F)
    } else {
      human <- normalise_text(human,removenumbers = F, stemDoc = F)
      human <- unlist(unique(human),use.names = F)
    }
    
    toms<- unlist(strsplit(machine,","))
    num<-sapply(toms, function(y) {
      grep(sprintf("\\b%s\\b",y),human,ignore.case =T)
    })
    
    if (noNA) {
      if (is.null(human)) {
        return(0.5)
        #Remove multiple as, it would be silly to match against the term "multiple"
      } else if (human == "" || human == "-" || lower =="not specified" || lower == "n/a"  ||
                 lower == "other/not specified" || lower == "not relevant" ||
                 lower == "no abstract" || lower == "n/a (chemo delivery system)" || 
                 lower == "n/a (imaging of mets)"|| lower == "n/a (imaging)" || 
                 lower =="n/a (imaging of mets) & imaging herceptin delivery"||
                 lower == "n/a (therapy delivery system)" || lower == "n/a surgery"|| 
                 lower == "none"|| lower =="not applicable" || 
                 lower == "0_no specific target" || human =="") {
        return(0.5)
      } else if (lower=="multiple" || lower == "other") {
        return(-1)
      }
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
  print(paste("# Multiple/Other:",sum(compare==-1),sep=" "))
  temp<-compare[compare!=0.5 & compare != -1]
  print(paste("accuracy without NA:",mean(temp>0),sep=" "))
}
