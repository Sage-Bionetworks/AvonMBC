synapseLogin()


results = synGet("syn11376396")

temp = synGet("syn11377989")



resultDf = read.csv(getFileLocation(results))
resultDf[colnames(grant.df)]
resultDf
