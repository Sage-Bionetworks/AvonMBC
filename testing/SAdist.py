import synapseclient
import os
import pandas as pd
import unicodedata
syn = synapseclient.login()


def removeascii(text):
    text[text.isnull()] = ''
    temp = []
    for i in text:
        if type(i) == unicode:
            temp.append(unicodedata.normalize('NFKD', i).encode('ascii','ignore'))
        else:
            temp.append(i)
    return(temp)

test_ent = syn.get("syn6172301")
test = pd.read_excel(test_ent.path)
test = test.apply(removeascii, axis=0)
testPath = test_ent.path.strip("xlsx")  + "csv"
test.to_csv(testPath,index=False)

SAdist_ent = syn.get("syn5587972")
SAdist = pd.read_excel(SAdist_ent.path)
SAdist = SAdist.apply(removeascii, axis=0)
SAdistPath = SAdist_ent.path.strip("xlsx") + "csv"
SAdist.to_csv(SAdistPath, index=False)


matchedSADistPath = "matchedSADist.csv"
os.system("Rscript ../R/clusterGrants.R %s %s %s" % (testPath, SAdistPath, matchedSADistPath))
matchedSADist = pd.read_csv(matchedSADistPath)
print(matchedSADist)
