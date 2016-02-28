library(synapseClient)
synapseLogin()
path = synGet("syn5574249")
load(path@filePath)

authors = read.csv("documents/SABCS_2008-2012_AUTHORS.csv",stringsAsFactors = F)
#grants = read.csv("documents/ICRP_allcombined_grants.csv",stringsAsFactors = F)
grants = read.csv("documents/MetCancer_Additional_Data_Feb2016_sub.csv", stringsAsFactors = F)
#15212 - 18322

matchingAuthors <- function(grants, authors) {
  abstractauthors = paste(authors$au_fname, authors$au_lname)
  grantauthors = paste(grants$PIFirstName, grants$PILastName)
  
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
  
  grant.df$SanAntonio_Abstracts[15212:18322] = unname(controls)
  save(grant.df,highlight.keywords,sanantonio, file = path@filePath)
}
