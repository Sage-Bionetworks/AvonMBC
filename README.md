# AvonMBC
###Matching annotations
**Matching_annotations.R**
Extracting all the human annotated terms for annotation category: Pathway, pathway group, molecular target, molecular target group, metastatic stage


###Normilization
**test_normalization.R**
```
library(tm)
library(RTextTools)
```
Create document term matrix and normalize grants - stem words, remove numbers, remove punctuation, remove sparse terms, remove stop words
Try [RTextTools](http://journal.r-project.org/archive/2013-1/collingwood-jurka-boydstun-etal.pdf) algorithms with "pathway" annotation
