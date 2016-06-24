library(parallel)
library(stringr)
require(Hmisc)
library(xlsx)
updateGeneList <- function(grantFile, geneFile) {
  grant.df <- read.xlsx(grantFile,sheetIndex = 1)
  #grant.df <- grantFile
  hgnc_genes <- read.csv(geneFile,sep="\t",stringsAsFactors = F)
  gene_info <- paste(hgnc_genes$symbol,hgnc_genes$alias_symbol,sep="|")
  gene_info <- strsplit(gene_info, split = "|",fixed=T)
  gene_info <- unlist(gene_info)
  gene_info <- unique(gene_info)

  combined <- paste(grant.df$AwardTitle, grant.df$TechAbstract,sep=" ")
  geneList <- sapply(seq_along(combined), function(i) {    
    text <- combined[i]
    #Split the word with word boundary then connect them back together with space
    words = str_split(text, boundary("word"))[[1]]
    temp = mclapply(gene_info, function(x) {
      if (length(grep(paste0("\\<", x,"\\>"),words)) >0) {
        x
      }
    })
    paste(unlist(temp),collapse="\n")

  })
  geneList
}