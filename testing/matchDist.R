
library(tm)
library(data.table)
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

makeDTM_abstract <- function(textVec){
  textVec <- strip.specialchar(textVec)
  textVec <- tolower(strip.markup(textVec))
  corpusraw <- Corpus(VectorSource(textVec)) 
  
  funs <- list(stripWhitespace,
               concatenate_and_split_hyphens,
               skipWords,
               removePunctuation,
               removeNumbers,
               stemDocument)
  
  corpus <- tm_map(corpusraw, FUN = tm_reduce, tmFuns = funs,mc.cores=1)
  
  
  ####Added this line to avoid error######
  corpus <- tm_map(corpus, PlainTextDocument)
  
  dtm <- DocumentTermMatrix(corpus, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))
  word.freq <- as.numeric(as.array(slam::rollup(dtm, 1, FUN=function(x) { sum(x > 0)})) )
  a=2
  dtm.sub <- dtm[, word.freq >a]
}



SAAbstracts <- "/home/tyu/.synapseCache/253/7330253/SABCS_2008-2012_ABSTRACTS.csv"
grantPath = "/gluster/home/tyu/.synapseCache/227/9720227/0csv"
distAuthors <- function(grantPath, SAAbstracts, SADistFile) {
  sanantonio <- fread(SAAbstracts,data.table=F)
  grant.df <- fread(grantPath,data.table=F)
  
  grant.MBC <- grant.df[grant.df$`Breast Cancer` >= 50,]
  rm(grant.df)
  text <- paste(grant.MBC$AwardTitle,grant.MBC$TechAbstract)
  textRownames <- paste0("MBC__",grant.MBC$AwardCode)
  rm(grant.MBC)
  sanantoniotext <- paste(sanantonio$title, sanantonio$body1)
  sanantonioRownames <- as.character(sanantonio$control)
  
  rm(sanantonio)
  distances <- lapply(seq_along(text), function(i) {
    final <- c(text[i], sanantoniotext)
    ff <- as.vector(final)
    #rm(final)
    ff<-toAmericanEnglish(ff)
    d <- makeDTM_abstract(ff)
    #rm(ff)
    temp<- as.matrix(d)
    row.names(temp) = c(textRownames[i],sanantonioRownames)
    temp <- t(temp)
    temp_cor <- cor(temp)
    
    paste(sanantonioRownames[order(temp_cor[,1][-1],decreasing = T)[1:20]],collapse=",")
  })
  write.csv(unlist(distances),file=SADistFile)
}





args <- commandArgs(trailingOnly = TRUE)
SADist <- distAuthors(args[1], args[2], args[3])