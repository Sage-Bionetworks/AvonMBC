library(synapseClient)
library(stringr)
require(Hmisc)

synapseLogin()

hgnc_genes <- read.csv("~/sandbox/hgnc_complete_set.txt",sep="\t",stringsAsFactors = F)
gene_info <- paste(hgnc_genes$symbol,hgnc_genes$name,hgnc_genes$alias_name, hgnc_genes$alias_symbol,sep="|")
gene_info <- strsplit(gene_info, split = "|",fixed=T)

all_grants <- read.csv("documents/ICRP_allcombined_grants.csv",stringsAsFactors = F)
combined = paste(all_grants$AwardTitle, all_grants$TechAbstract,sep=" ")

final_gene_info = sapply(gene_info, function(x) {
  genes = unlist(x)
  genes = genes[genes!=""]
  genes = sub("[[]","",genes)
  genes = sub("]","",genes)
  genes = str_split(genes, boundary("word"))
  capped = lapply(genes, capitalize)
  total = c(genes, capped)
  final = lapply(total, function(y) {
    if(length(y)>1) {
      return(paste(y, collapse = " "))
    } else {
      return(y)
    }
  })
  return(unlist(final))
})

rm(gene_info)

geneList <- sapply(seq_along(combined)[1:2], function(i) {    
  text <- combined[i]
  xtime <- Sys.time()
  #Split the word with word boundary then connect them back together with space
  words = str_split(text, boundary("word"))
  words = paste(c(" ",unlist(words)),collapse= " ")
  print("words")
  print(Sys.time()-xtime)
  temp = sapply(final_gene_info, function(x) {
    count = sapply(x, function(y) {
      str_detect(paste0(" ", y, " "),words)
    })
    sign(sum(count)>0)
  })
  print("genes")
  print(Sys.time()-xtime)
  indexes <- which(temp==1)
  main = paste(hgnc_genes$symbol[indexes],hgnc_genes$name[indexes],sep=": ")
  total = paste(main,hgnc_genes$alias_symbol[indexes],sep=" ---- ")
  print(i)
})
