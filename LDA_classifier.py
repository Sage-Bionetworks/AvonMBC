from openpyxl import load_workbook
import csv, json, textmining, os



dat_mbc = load_workbook('MetastaticBC_Grants_CodingFile_Complete_Sagebase.xlsx')
#print dat_mbc.get_sheet_names()
dat_grant = dat_mbc['Data']
#print(dat_grant['A1'].value)

#pip install textmining
#the bottom here, the class is defined
tdm_title = textmining.TermDocumentMatrix(textmining.simple_tokenize_remove_stopwords)
tdm_abstract = textmining.TermDocumentMatrix(textmining.simple_tokenize_remove_stopwords)

header = [val.value.strip() for val in list(dat_grant.rows[0]) if val.value]
idx = 1
for row in dat_grant.rows[1:]:
  value = [val.value for val in list(row)] # allow NULL field here, and truncate later
  grant_one = {"ID":idx}
  for field, val in zip(header, value):
    print val
  tdm_title.add_doc(grant_one["AwardTitle"])
  tdm_abstract.add_doc(grant_one["TechAbstract"])


model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(tdm_abstract)
topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
	print('Topic {}: {}'.format(i, ' '.join(topic_words)))