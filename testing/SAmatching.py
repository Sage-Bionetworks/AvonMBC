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
testPath = test_ent.strip("xlsx")  + "csv"
test.to_csv(testPath,index=False)

SAdist_ent = syn.get("syn5587972")
SAdist = pd.read_excel(SAdist_ent.path)
SAdist = SAdist.apply(removeascii, axis=0)
SAdistPath = SAdist_ent.path.strip("xlsx") + "csv"
SAdist.to_csv(SAdistPath, index=False)
matchedSAAuthorPath = "matchedSAAuthor.csv"
os.system("Rscript ../R/matching_SanAntonio_authors.R %s %s %s" % (test_ent.path, SAauthors_ent.path, matchedSAAuthorPath) )
matched_SAauthors = pd.read_csv(matchedSAAuthorPath)
print(matched_SAauthors)
