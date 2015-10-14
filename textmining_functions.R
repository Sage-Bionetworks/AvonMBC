library(tm)

skipWords <- function(x) removeWords(x, c(stopwords('english')))
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
replace.greek <- function(x){
  for(i in 1:length(greekSubs)){x <- gsub(greekSubs[[i]], names(greekSubs)[i], x)}
  return (x)
}
strip.markup <- function(x){gsub("</?[A-Za-z]+>|<[A-Za-z]+\\s?/>"," ", x)}
strip.specialchar <- function(x){gsub("&#\\d+;"," ", iconv(x, "latin1", "UTF-8"))}
prepareText <- function(x){concatenate_and_split_hyphens(removepvalue(tolower(strip.specialchar(replace.greek(strip.markup(x))))))}

toAmericanEnglish <- function(x){
  gsub("tumour","tumor",x)
}
cleanFun <- function(htmlString) {
  return(gsub("<.*?>", "", htmlString))
}

normalise_text <- function(textVec) {
  textVec <- tolower(strip.markup(textVec))
  corpusraw <- Corpus(VectorSource(textVec)) 
  
  funs <- list(stripWhitespace,
               concatenate_and_split_hyphens,
               skipWords,
               removePunctuation,
               removeNumbers,
               stemDocument)
  #normalize
  corpus <- tm_map(corpusraw, FUN = tm_reduce, tmFuns = funs)
  
  ####Added this line to avoid error######
  corpus <- tm_map(corpus, PlainTextDocument)

  #Convert corpus to dataframe  
  dataframe<-data.frame(abstracts=unlist(sapply(corpus, `[`, "content")), 
                        stringsAsFactors=F)
  return(dataframe)
}