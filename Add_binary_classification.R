library(dplyr)
library(data.table)

#-----------------------------------------
#Create Binary Columns for each category
#Remove NA', other??
#-----------------------------------------
create_binary <- function(needed,group) {
  temp<-unlist(needed[group],use.names = F)
  #Filter out NA's first
  temp[is.na(temp)]<-0
  lower <- tolower(temp)
  #Make not specified = 0
  temp[which(lower == "not specified" | 
               lower == "n/a" | lower == "-" |
               lower == "other/not specified" | lower == "not relevant" | 
               lower == "no abstract" | lower == "n/a (chemo delivery system)" | 
               lower == "n/a (imaging of mets)"| lower == "n/a (imaging)" | 
               lower =="n/a (imaging of mets) & imaging herceptin delivery"|
               lower == "n/a (therapy delivery system)" | lower == "n/a surgery"| 
               lower == "none"| lower =="not applicable" | 
               lower == "0_no specific target")] <- 0
  temp[which(temp!=0)]<-1
  #Must have global variable (needed)
  needed[group]<- temp
  return(needed)
}
#----------------------------------------
#Workflow
#----------------------------------------
grantInfo<-read.csv("Metastatic_grant.csv",stringsAsFactors=F)
#Only get needed info
neededInfo <- grantInfo[,c(2:17)]
#Dplyr dataframe
needed <- tbl_df(neededInfo)
#Add columns
needed<-mutate(needed, pw_binary = Pathway, pwgroup_binary = Pathway..Group., mt_binary = Molecular.Target,
               mtgroup_binary = Molecular.Target..Group., meta_binary = X.Metastasis.stage)

needed <- create_binary(needed,"pw_binary")
needed <- create_binary(needed,"pwgroup_binary")
needed <- create_binary(needed,"mtgroup_binary")
needed <- create_binary(needed,"mt_binary")
needed <- create_binary(needed,"meta_binary")

write.csv(needed,"metastatic_grant_binary.csv")
t<-fread("metastatic_grant_binary.csv",data.table = F)


