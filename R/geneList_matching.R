library(parallel)
library(stringr)
require(Hmisc)
library(xlsx)
#library(synapseClient)


updateGeneList <- function(grantFile, geneFile, geneFileName) {
  grant.df <- read.xlsx(grantFile,sheetIndex = 1)
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
    },mc.cores=3)
    paste(unlist(temp),collapse="\n")

  })
  write.csv(geneList, file=geneFileName)
  geneList
}

#synapseLogin()
#geneList_ent <- synGet("syn5594707")
#test_ent = synGet("syn6172301")
args <- commandArgs(trailingOnly = TRUE)
geneList <- updateGeneList(args[1], args[2], args[3])
