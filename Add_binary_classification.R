library(dplyr)
grantInfo<-read.csv("Metastatic_grant.csv",stringsAsFactors=F)
#Only get needed info
neededInfo <- grantInfo[,c(2:17)]
#Dplyr dataframe
needed <- tbl_df(neededInfo)
#Add column
needed<-mutate(needed, pw_binary = Pathway)
#Remove NA', other??
needed$Pathway[which(is.na(needed$pw_binary) | 
                         tolower(needed$pw_binary) == "not specified" | 
                         tolower(needed$pw_binary) == "n/a" | 
                         needed$pw_binary == "-")] <- 0
unique(needed$Pathway)

