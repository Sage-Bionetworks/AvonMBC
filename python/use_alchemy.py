#python alchemyapi.py 9d3dfcf3fe8b276a44952fb19a85d593bf1b0e88
import nltk
from 3rd_party_software/alchemyapi import AlchemyAPI
import pandas as pd
alchemyapi = AlchemyAPI()

grants = pd.read_csv("metastatic_grants_binary")
test = grants['TechAbstract']
alchemyapi.entities('html',test[1])
alchemyapi.keywords('html', test[1])
alchemyapi.concepts('html',test[1])
pathway = grants['Pathway']


subset = test[1:100]

for i in range(100):
	keywords = alchemyapi.keywords('html', subset[i])
	pw = pathway[i]
	words = [i['text'] for i in keywords['keywords']]
	#paragraph = " ".join(words)
	#tokens = nltk.word_tokenize(paragraph)
	list(pw = pw,keywords= words)
