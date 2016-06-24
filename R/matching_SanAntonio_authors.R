library(xlsx)
#authors = read.csv("documents/SABCS_2008-2012_AUTHORS.csv",stringsAsFactors = F)
matchingAuthors <- function(grantPath, authorsPath, SAAuthorFile) {
  authors <- read.xlsx(authorsPath,sheetIndex=1)
  grants <- read.xlsx(grantPath,sheetIndex = 1)
  abstractauthors <- paste(authors$au_fname, authors$au_lname)
  grantauthors <- paste(grants$PIFirstName, grants$PILastName)
  
  abstractauthors <- gsub("[[:punct:]]","",abstractauthors)
  grantauthors <- gsub("[[:punct:]]","",grantauthors)
  
  controls <- sapply(grantauthors, function(x) {
    name = sapply(unlist(strsplit(x," ")), function(word) {
      if(nchar(word)!=1)
        return(word)
      return(NULL)
    })
    x <- sub("NULL","", paste(name,collapse=" "))
    paste(authors$control[tolower(abstractauthors) %in% tolower(x)],collapse = ",")
  })
  rm(authors)
  rm(grants)
  write.csv(unname(controls),file=SAAuthorFile)
}

args <- commandArgs(trailingOnly = TRUE)
geneList <- matchingAuthors(args[1], args[2], args[3])
