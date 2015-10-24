#Creates features
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import svm

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
#NLTK -> tokenizer, stemmer
import nltk
from nltk.collocations import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
nltk.download("stopwords")
import pandas
import string
import re
#Beautiful soup -> Gets rid of HTML and replaces non ascii characters
pip install Beautifulsoup4
from bs4 import BeautifulSoup
import unicodedata 
import numpy


#-------------------------------------------------------------------------------
#Function tha checks if it is a number, if not return the word, if it is return a stop word
#-------------------------------------------------------------------------------
def is_number(s):
    try:
        float(s)
        return 'the'
    except ValueError:
        return s

#--------------------------------------------------------------------------------------
#Normalization of text (Remove HTML, replace non-ascii characters, remove punctuation,
#remove lone numbers, stem words, remove stop words recombine into text)
#--------------------------------------------------------------------------------------
#grantInfo = pandas.read_csv('metastatic_grant_binary.csv')
#total = grants['AwardTitle'] + ", " + grants['TechAbstract']

#total -> A list of text (there are two text right now)
total = ['Yin and Yang of Heparanase in Breast Cancer Initiation, Background and Significance:  Heparanase-1 (HPR1) is an endoglycosidase overexpressed in many malignancies including breast cancer.  HPR1 degrades heparan sulfate (HS) proteoglycans (HSPG), the main component of the cell surface, basement membrane (BM), and extracellular matrix (ECM).  Breakdown of HSPG by HPR1 leads to release of growth factors trapped in the BM and ECM.  Numerous studies using xenograft models have confirmed the role of HPR1 in tumor angiogenesis and metastasis.  However, several lines of evidence suggest that HPR1 may exert its angiogenic effect independent of its enzymatic activity: (1) HPR1 induces endothelial cell migration by activating Akt through its LRP receptor independent of its enzymatic activity; (2) exogenous HPR1 or increased HPR1 expression in HPR1-transfected endothelial cells can induce the expression of VEGF in HPR1 enzymatic activity-independent manner; and (3) the C-terminus of HPR1 in the absence of HPR1 enzymatic activity can stimulate a stronger tumor angiogenesis than the enzymatically active HPR1.  Emerging evidence from our laboratory and others suggests that HPR1 enzymatic activity may be not only unnecessary for its antiogenic function but rather has a negative effect on tumor growth: (1) HS degradation on the cell surface by HPR1 can attenuate the signaling activity mediated by growth factors such as FGF2 that require HSPGs as their co-receptors.  (2) MMTV-Wnt transgenic mice on a HSPG-null background (syndecan-1 knockout) become resistant to breast cancer formation.  (3) Syndecan-1 and glypican-1, two HSPGs overexpressed in breast cancer, can promote breast cancer development.  (4) Using a recently reported, clinically relevant somatic mouse breast cancer model, our preliminary study showed that treatment with sulodexide, a low-molecular-weight heparin and a potent HPR1 inhibitor, accelerated breast tumor formation induced by polyoma virus middle T antigen (PyMT).<br /><br />Objective/Hypothesis:  Based on these new data, we hypothesize that HPR1 enzymatic activity can antagonize the tumor-promoting effect of the C-terminus of HPR1.  The goal of this proposal is to dissect the opposing effect of HPR1 enzymatic activity and HPR1 C-terminus epitope on breast tumor initiation in a clinically relevant mouse breast cancer model.<br /><br />Specific Aims:  (1) To determine if HPR1 knockdown will suppress or accelerate breast tumor initiation mediated by three oncogenes: PyMT, Neu, and Wnt.  (2) To determine whether HPR1 C-terminus or an enzymatically dead HPR1 stimulates breast tumor initiation, whereas full-length HPR1 has no effect or is less effective in stimulating breast tumor initiation and progression.<br /><br />Research Strategy:  (1) The ability of a dual microRNA sequence that targets mouse HPR1 (mHPR1) in murine breast cancer cell lines to suppress HPR1 expression has been verified.  This mHPR1 microRNA insert will be inserted downstream of PyMT, Neu, or Wnt oncogene cloned in the avian retroviral vector (RCAS).  A control vector carrying a scrambled miRNA sequence will be included as a negative control.  These retroviral vectors will be used to induce breast cancer by intraductal injection into the mammary gland of the transgenic mice carrying the transgene encoding the receptor for the subgroup A avian leucosis virus.  If suppression of HPR1 gene expression can accelerate breast cancer initiation, it will suggest that mHPR1 has an overall suppressive effect on breast cancer initiation.  (2) RCAS vectors carrying the C-terminus of HPR1, the full-length HPR1, or a mutated HPR1 (enzymatic activity-dead) gene will be prepared and used to infect TVA transgenic mice intercrossed with MMTV-PyMT, MMTV-Neu, or MMTV-Wnt transgenic mice.  If HPR1 C-terminus and mutated HPR1 gene (no HPR1 activity) are more potent at accelerating breast cancer initiation than the full-length of HPR1, it will suggest that HPR1 enzymatic activity has a negative effect on breast tumor initiation.<br /><br />Impact:  These studies will greatly improve our understanding of the complex functions of HRP1 and allow better design of therapeutic strategy for treating breast cancer.<br /><br />Innovation:  We have proposed a novel hypothesis that HPR1 enzymatic activity can antagonize the tumor-promoting effect of the C-terminus of HPR1.  This hypothesis will be tested in a unique mouse model.  Results from these studies will challenge the current paradigm that HPR1 enzymatic activity can serve as a molecular target for antitumor therapy.  In addition, our study to address the role of HRP1 C-terminus in breast tumor initiation is an underexplored area.<br />', 'Wake up of dormant metastatic breast cancer cells by radiation: prevention with a COX-2 inhibitor, After arriving in a secondary site, metastatic cells may begin proliferate, undergo apoptosis (a type of cell death) or remain as dormant cells. To prevent the occurrence of metastases in distant organs, different therapies (e.g. chemotherapy) are currently administered. However, metastatic dormant cells are resistant to conventional therapies that target actively dividing cells. This resistance likely account for disease recurrence since it is highly probable that dormant cancer cells will ultimately be the source of subsequent overt metastatic disease years to decades after primary tumour diagnosis. By definition, dormancy implies the ability to be awoken. Coupled closely to this question is what awakes dormant tumour cells? Radiation increases the activity of cyclooxygenase-2 (COX-2) which produces the prostaglandin E2 (PGE2). Since PGE2 can stimulate the proliferation of cancer cells, we propose that irradiation of a breast tumour and surrounding tissues could induce a switch from dormant cancer cells to proliferative cells. The mouse mammary cancer cell D2.0R will be used as model of cancer cell dormancy. The D2.0R cells remain cell cycle arrested in a three-dimensional (3D) culture in vitro, and after injection in the tail vein of a mouse they can migrate to the lungs where they form dormant metastases. We will determine whether irradiation of cancer cells or fibroblasts can induce a switch from dormant cancer cells to proliferative cells in a 3D cell culture system and in the mouse model. Then, the capacity of a COX-2 inhibitor to prevent the wakeup of dormant cancer cells will be assessed.']
stopwords.words('english')

#Remove HTML, replace non-ascii characters, tokenize (remove punctuation), stem words, then reconcatenate
text_noHTML = [BeautifulSoup(i).get_text() for i in total]
total_ascii = [unicodedata.normalize('NFKD', text).encode('ascii','ignore') for text in text_noHTML]
total_noperiod = [re.sub(r'\D\.\D',' ', text) for text in total_ascii]
total_noslash = [re.sub(r'\b\/\b',' ', text) for text in total_noperiod]
total_combinedash = [re.sub(r'\b-\b',"",text) for text in total_noslash]
total_nopunct = [nltk.word_tokenize(text.translate(None, string.punctuation)) for text in total_combinedash]

#Use english stemmer, remove lone numbers, and join back into a paragraph of normalized words
stemmer = SnowballStemmer("english")
total_complete[:]
for i in range(0,len(total_nopunct)):
	temp  = [is_number(text)  for text in total_nopunct[i] ]
	temp = [stemmer.stem(word) for word in temp]
	temp = [word for word in temp if word not in stopwords.words('english')]
	total_complete[i] = ' '.join(temp)


#----------------------------------------------------------------
# SCIKIT TOOLS -> Get features by using TfidF Vectorizer
# Change text into binary matrix with respect to features
#----------------------------------------------------------------

#Vectorizer, with no tfidf
vectorizer = CountVectorizer(max_df=0.0005,stop_words='english')
X = vectorizer.fit_transform(total_complete)
vectorizer.get_feature_names()
features_array = X.toarray()

#Vectorizer with Tfidf
vectorizer = TfidfVectorizer(max_df=0.0005,stop_words='english')
X = vectorizer.fit_transform(total_complete)
vectorizer.get_feature_names()
features_array = X.toarray()
