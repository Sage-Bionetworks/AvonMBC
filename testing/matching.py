import synapseclient

import pandas as pd
# R files
import rpy2.robjects as robjects

def geneList_matching(testPath, geneListPath):
    robjects.r("source('../R/geneList_matching.R')")
    updateGeneList = robjects.r('updateGeneList')
    updateGeneList(testPath, geneListPath)


syn = synapseclient.login()
geneList_ent = syn.get("syn5594707")
test_ent = syn.get("syn6172301")
matched_genes = geneList_matching(test_ent.path, geneList_ent.path)
print(matched_genes)


