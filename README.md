# AvonMBC
###Matching annotations
**Matching_annotations.R**

Extracting all the human annotated terms for annotation category: 
* Pathway 
* Pathway Group
* Molecular Target
* Molecular Target Group
* Metastatic Stage. 

Parsing through the extracted human annotated terms for each category
-Extract words in ()
-Remove and,/,or,-,a,n,K,3,-1,-2,""

Match extracted terms to grant abstract and title

Compare these matched terms to the actual annotations

###Normilization
**test_normalization.R**
```
library(tm)
library(RTextTools)
```
Create document term matrix and normalize grants:
-stem words
-remove numbers
-remove punctuation
-remove sparse terms
-remove stop words

Try [RTextTools](http://journal.r-project.org/archive/2013-1/collingwood-jurka-boydstun-etal.pdf) algorithms with "pathway" annotation
