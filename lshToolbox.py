# -*- coding: utf-8 -*-
"""
@author: Roberto Lucchese, Stefano Cecconello and Thierry van der Spek

"""
import numpy as np
import md5
import string
import itertools 
import random

"""
The Jaccard coefficient measures similarity between finite sample sets,
and is defined as the size of the intersection divided by 
the size of the union of the sample sets
"""

def jsim(s1,s2):
    return len(s1 & s2) / float (len(s1 | s2))

"""
minHash with Permutations() 
S = sets
k = number of permutations
seed = np.random.seed(seed)
"""
def minhash_Perm(S,k,seed): 
   np.random.seed(seed);

   U = set().union(*S); #union of the Sets
   numbSets = len(S); #total number of Sets
   lu = len(U); #amount of the unique elements in the union

   U = sorted(U); #Sorting U
   #print  'Sorted list:',U; #printing U after sorting

   M = np.zeros((lu, numbSets), dtype=np.int); #inizitializing the Matrix Representation of Sets
   SignMP = np.zeros((k, numbSets),dtype=np.int); #initializing the Signature Matrix
   
   #Building the Matrix Representation of Sets
   for r in range(lu): #from row 0 to row = number of unique elements
       for c in range(numbSets): #from column 0 to column = number of Sets?Docs
          if(U[r] in S[c]):
             M[r][c] = 1;   
             
   
   
   #CREATION OF THE SIGNATURES
   i = 1 #number useful for the signature
   #create k permutations of size lu
   for k_ind in range(k):
    #perm = np.array([1,3,7,6,2,5,4]) #First permutation slide
    #perm = np.array([4,2,1,3,6,7,5]) #Second permutation slide
    #perm = np.array([3,4,7,6,1,2,5]) #Third permutation slide
    perm = np.random.permutation(range(lu)) 
    lstPerm = list(perm);
    for c in range(numbSets): #for each column 
        i =0
        for r in range(lu): #for each row
            itemindex = lstPerm.index(i); #takes the index of i in the permutation
            if(M[itemindex][c] == 1): #if in the input matrix is 1 
                SignMP[k_ind][c] = i+1; #put in the signature matrix[k_ind][c] = i
                break;
            else:
                i = i+1; #otherwise increment i
                  
   return [U,SignMP]  

"""
Generate hash function from MD5 family
"""
def hash_md5(n):
   random.seed(n)
   rs = ''.join(random.choice(string.lowercase) for i in range(64))
   m = md5.new()

   def myhash(x):
     m.update(str(x)+rs)
     hs = m.hexdigest()
     return int(hs, 32)

   return myhash    
      
"""
minHash with Hash Functions() 
S = sets
k = number of permutations
seed = np.random.seed(seed)
We use the md5() family
"""   
def minhash_Hash(S,k,seed): 
   np.random.seed(seed);
   
   U = sorted(set().union(*S)); #union of the Sets and sorting
   numbSets = len(S); #total number of Sets
   lu = len(U); #amount of the unique elements in the union
  
   
   M = np.zeros((lu, numbSets)); #inizitializing the Matrix Representation of Sets
   SignM = np.zeros((k, numbSets)); #initializing the Signature Matrix
   
   for r in range(k): #from row 0 to row = number of unique elements
       for c in range(numbSets): #from column 0 to column = number of Sets?Docs
            SignM[r][c] = float("inf")
   
   #Building the Matrix Representation of Sets
   for r in range(lu): #from row 0 to row = number of unique elements
       for c in range(numbSets): #from column 0 to column = number of Sets?Docs
          if(U[r] in S[c]):
             M[r][c] = 1;  
   
   #MAIN METHOD
   for r in range(lu): #from row 0 to row = number of unique elements
       for c in range(numbSets): #from column 0 to column = number of Sets?Docs
           if(M[r][c] == 1):
               for i in range(k):
                   hf = hash_md5(i)
                   if(hf(r) <  SignM[i][c]):
                       SignM[i][c] = hf(r)
    

   return [U,SignM]   
   
"""
sissim(ss1,ss2) 
ss1 = signature 1
ss2 = signature 2

"""
def sigsim(ss1,ss2):
    if len(ss1) != len(ss2):
        return False
    cont = 0
    for i in range(len(ss1)):
        if ss1[i] == ss2[i]:
            cont += 1

    return cont / float(len(ss1))
    
"""
simmat(M)
M = matrix of column vector
If M = N1xN2 return the a matrix N2xN2 with the similarity of each column of M
"""
def simmat(M):
   #transpose the matrix in order to take the row vectors
   M = M.T;
   TotSimMatrix = np.zeros((M.shape[0],M.shape[0]));

   for row1 in range(M.shape[0]):
       for row2 in range(M.shape[0]):
           TotSimMatrix[row1,row2] = sigsim(M[row1],M[row2])
   return TotSimMatrix  
  
"""
Find all subsets of S of lenght m
"""  
def allsubsets(S, m):
   
    return set(itertools.combinations(S, m))   
    
"""
Jaccard Similarity between all subsets (in pairs)
"""          
def jSimAll(subset):
    jsim_array = np.zeros((len(subset), len(subset)))

    for i in range(len(subset)):
        for j in range(i, len(subset)):
            jsim_array[i][j] = jsim(subset[i], subset[j]);
            jsim_array[j][i] = jsim(subset[j], subset[i]);
    return jsim_array          


"""
optimumRange Task 4
"""
def optimumRange(avg, value):
    if (avg + 0.02) >= value and value >= (avg - 0.02):
        return True
    else:
        return False

"""
optimumRange Task 7
"""
def optimumRange_Task7(avg, value):
    if (avg + 0.05) >= value and value >= (avg - 0.05):
        return True
    else:
        return False

"""
Divide in bands  
"""
def banding(initialMat, band, rows):
    singol = [[0 for r in range(rows)] for b in range(band)]
    for k, v in enumerate(initialMat):
        singol[k / rows][k % rows] = v
    return singol
 
"""
Banding similarity
finalBanding[i][j] = 1 if the signatures S(i) and S(j) have at least one identical bands
"""
def bsim(Minp, band, rows, th=0):
    mat = Minp.T
    finalBanding=np.zeros([len(mat),len(mat)])

    for i in xrange(len(finalBanding)):
        for j in xrange(i, len(finalBanding)):
            if i == j:
                finalBanding[i][j] = 1
            else:
                eqbands = 0
                bands1 = banding(mat[i], band, rows)
                bands2 = banding(mat[j], band, rows)
                for ix in xrange(len(bands1)):
                    if bands1[ix] == bands2[ix]:
                        eqbands += 1
                if eqbands > 0:
                    finalBanding[i][j] = 1
                    finalBanding[j][i] = 1

    return finalBanding


"""
Compute the Cosine Similarity based on the angular distance between vectors

"""
def cossim(a,b):
    
    if np.array_equal(a, b):
        return 1.0
    else:
        return 1 - np.degrees(np.arccos(np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))))/180


"""
#k = number of random directions (not -1,+1 as shown in the book but normally 
distributed random numbers)

#Vk = normally distributed random vectors (v1,v2,v3 shown in the book)

#Sketches = Matrix of -1 and +1, which consist of column vectors of -1's and 
+1's of length k that represent sketches of column vectors from M

#M = array of doubles where each column represent a point in d dimensional space
with d>1
"""

def sketch(M,k):
   
  M = M.T #before M had row vectors and not column vectors 
  #print '\n Transposed matrix:\n ',M
  shape = M.shape #Shape of the matrix M
  
 
  n = shape[0] #number of rows of M (M was transposed) = length of each vector     
  d = shape[1] #dimension d = number of column of the matrix M (M was transposed) 
  
  
  #print 'Dimension d =', d;
  #print 'Number of rows n =', n;
  
  sum_vx = 0;
  
  Vk = np.zeros((k,n))  #initializing k vectors with n elements each to zero
  
  for i in range(k): #generating k vectors of normally distributed random numbers
   Vk[i] = np.random.standard_normal(n) 
   #print 'Random direction ',i,' = ',Vk[i]
   
   #vector column of sketches
   Sketches = np.zeros((k,d))  #initializing k vectors with d elements each to zero
  
  #Vk = np.array([[1,-1, 1, 1],[-1,1,-1,1],[1,1,-1,-1]]); 
  #print 'Directions as the book\n',Vk
   
   #Sketches generator
  for s in range(d): #scan each column of M
   for q in range(k): #from 1 to number of vectors Vk
       sum_vx=0  
       for r in range (n): #scan each row of M
            sum_vx += M[r][s]*Vk[q][r]   
       if(sum_vx>0):
            Sketches[q][s] = 1
       else:
            Sketches[q][s] = -1
  
  return Sketches 

"""
Sketch for the main task.. 
M[r,s] instead of M[r][s] 
"""
def sketch_main(M,k):
   
  M = M.T #before M had row vectors and not column vectors 
  #print '\n Transposed matrix:\n ',M
  shape = M.shape #Shape of the matrix M
  
 
  n = shape[0] #number of rows of M (M was transposed) = length of each vector     
  d = shape[1] #dimension d = number of column of the matrix M (M was transposed) 
  
  
  #print 'Dimension d =', d;
  #print 'Number of rows n =', n;
  
  sum_vx = 0;
  
  Vk = np.zeros((k,n))  #initializing k vectors with n elements each to zero
  
  for i in range(k): #generating k vectors of normally distributed random numbers
   Vk[i] = np.random.standard_normal(n) 
   #print 'Random direction ',i,' = ',Vk[i]
   
   #vector column of sketches
   Sketches = np.zeros((k,d))  #initializing k vectors with d elements each to zero
  
  #Vk = np.array([[1,-1, 1, 1],[-1,1,-1,1],[1,1,-1,-1]]); 
  #print 'Directions as the book\n',Vk
   
   #Sketches generator
  for s in range(d): #scan each column of M  
   for q in range(k): #from 1 to number of vectors Vk
       sum_vx=0  
       for r in range (n): #scan each row of M
            sum_vx += M[r,s]*Vk[q][r] 
       if(sum_vx>0):
            Sketches[q][s] = 1
       else:
            Sketches[q][s] = -1
  
  return Sketches 
