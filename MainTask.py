"""
@author: Roberto Lucchese, Stefano Cecconello and Thierry van der Spek

"""

from lshToolbox import cossim,bsim,sketch_main
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


categories = ['alt.atheism',
 'comp.graphics']
# 'comp.os.ms-windows.misc',
# 'comp.sys.ibm.pc.hardware',
# 'comp.sys.mac.hardware',
# 'comp.windows.x',
# 'misc.forsale',
# 'rec.autos',
# 'rec.motorcycles',
# 'rec.sport.baseball']
 
#newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_train = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'),categories=categories)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectorFull = csr_matrix.todense(vectors) #from sparse matrix to dense matrix
documentNumber=newsgroups_train.target.shape[0]
categoryList = [[],[],[],[],[],[],[],[],[],[]]
for i in range(documentNumber):
    currentCategory=newsgroups_train.target[i]
    categoryList[currentCategory].append(vectorFull[i])
print 'the num of documents is'
print newsgroups_train.target.shape[0]
"call the function Sketch on vectorFull and k directions"
print time.time()
print 'Start sketch ...'
sketchMat = sketch_main(vectorFull,30)
print 'End sketch '
print time.time()
print 'SketchMat dimension (rows,column) :',sketchMat.shape
band=5
rows=6

#The final matrix with the number of similirity per category
matrix=np.zeros([len(categories),len(categories)])
matrixElements=np.zeros([len(categories),len(categories)])
print 'Start banding ...'
bands = bsim(sketchMat, band, rows)
print 'End banding'

threshold=(1/band)**(1/rows)
print 'Starting to compute the final matrix ...'
for i in range(len(bands)):
    for j in range(len(bands)):
        #If the two elements have at least one bucket in which collide
        if bands[i][j]==1:
            #calculating the cosine similarity of the two sketch and see if it exceeds the threshold
            if cossim(sketchMat[:,i],sketchMat[:,j])>=threshold:
                #find the categories that are related to the sketch and do +1 on the element of the final matrix that corresponds to that 2 categories
                cat1=newsgroups_train.target[i]
                cat2=newsgroups_train.target[j]
                matrix[cat1,cat2]=matrix[cat1,cat2]+1


for i in range(len(categories)):
    for j in range(len(categories)):
        #the number of elements of the two categories that we compare and pass the threshold 
        #divided by the number of pairs that are formed between the two categories
         matrix[i,j]=matrix[i,j]/float((len(categoryList[i])*len(categoryList[j])))
print matrix


