
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


# In[36]:


# read in the dataset
df = pd.read_csv('train.data', sep='\n', header=None)
tdf = pd.read_csv('test.data', sep='\n', header=None)

# separate names from classes
vals = df.iloc[:,:].values
tvals = tdf.iloc[:,:].values

# print(vals)

docs = []

cls = []

test_docs = []

for i in vals:
    cls.append(i[0][:1])
    docs.append(i[0][2:])

for i in tvals:
    test_docs.append(i[0])

print(len(vals))
    
print("descriptions - ",len(docs))

print("cls - ",len(cls))

print("test_descriptions - ",len(test_docs))

### FILL IN THE BLANKS ###


# In[37]:


c1,c2,c3,c4,c5 = 0,0,0,0,0
tcls = [int(i) for i in cls]
for i in tcls:
    if i==1:
        c1+=1
        continue
    if i==2:
        c2+=1
        continue
    if i==3:
        c3+=1
        continue
    if i==4:
        c4+=1
        continue
    if i==5:
        c5+=1
print("c1 - c2 - c3 - c4 - c5 \n",c1,c2,c3,c4,c5)
print("Total classes - ", c1+c2+c3+c4+c5)


# In[38]:


print("\nTraining Doc Sample - ",docs[:1])
print("\n\nTesting Doc Sample - ",test_docs[:1],"\n")


# In[39]:


tcls = np.asarray(tcls)
print(type(docs))
print(type(tcls))
docs = [d.split() for d in docs]
test_docs = [td.split() for td in test_docs]


# In[40]:


def filterLen(docs, minlen):
    r""" filter out terms that are too short. 
    docs is a list of lists, each inner list is a document represented as a list of words
    minlen is the minimum length of the word to keep
    """
    return [ [t for t in d if len(t) >= minlen ] for d in docs ]
docs = filterLen(docs, 4)
test_docs = filterLen(test_docs, 4)

temp_docs = []

for doc in docs:
    temp_doc = []
    for word in doc:
        temp = ''.join(c for c in word if c.isalnum())
        temp_doc.append(temp.lower())
    temp_docs.append(temp_doc)

docs = temp_docs

temp_docs = []

for doc in test_docs:
    temp_doc = []
    for word in doc:
        temp = ''.join(c for c in word if c.isalnum())
        temp_doc.append(temp.lower())
    temp_docs.append(temp_doc)

test_docs = temp_docs
    
print(len(docs[0]), docs[0][:20])
# print(len(docs1[0]), docs1[0][:20])
print(len(test_docs[0]), test_docs[0][:20])
# print(len(test_docs1[0]), test_docs1[0][:20])


# In[41]:


from collections import Counter
from scipy.sparse import csr_matrix
def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat, idx

def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )


# In[42]:


train_mat, idx = build_matrix(docs)
csr_info(train_mat,non_empy=True)


# In[43]:


def build_matrix_test(docs,idx):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
#     idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
#         for w in d:
#             if w not in idx:
#                 idx[w] = tid
#                 tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            temp = idx.get(k,-1)
            if temp != -1:
                ind[j+n] = temp
                val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat


# In[44]:


test_mat = build_matrix_test(test_docs,idx)
csr_info(test_mat,non_empy=True)


# In[45]:


# scale matrix and normalize its rows
def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat
train_mat1 = csr_idf(train_mat, copy=True)
train_mat2 = csr_l2normalize(train_mat1, copy=True)
print("train_mat:", train_mat[15,:20].todense(), "\n")
print("train_mat1:", train_mat1[15,:20].todense(), "\n")
print("train_mat2:", train_mat2[15,:20].todense(),"\n")
print(train_mat2.shape)
print(train_mat2.shape[0])
print(train_mat2.shape[1])


# In[46]:


test_mat1 = csr_idf(test_mat, copy=True)
test_mat2 = csr_l2normalize(test_mat1, copy=True)
print("test_mat:", test_mat[15,:20].todense(), "\n")
print("test_mat1:", test_mat1[15,:20].todense(), "\n")
print("test_mat2:", test_mat2[15,:20].todense(),"\n")
print(test_mat2.shape)
print(test_mat2.shape[0])
print(test_mat2.shape[1])


# In[47]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_mat2, tcls)
predicted = clf.predict(test_mat2)
print(predicted[:5])
with open('predictions_1.data','w+') as file:
    for p in predicted:
        file.write(str(p)+"\n")


# In[49]:


from sklearn.linear_model import SGDClassifier
clf_2 = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=10, tol=None).fit(train_mat2, tcls)
predicted_2 = clf_2.predict(test_mat2)
print(predicted_2[:5])
with open('predictions_2.dat','w+') as file:
    for p in predicted_2:
        file.write(str(p)+"\n")


# In[16]:


# from sklearn.model_selection import GridSearchCV
# parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
#               'tfidf__use_idf': (True, False),
#               'clf__alpha': (1e-2, 1e-3),
# }
# gs_clf = GridSearchCV(clf_2, parameters, n_jobs=-1)
# gs_clf = gs_clf.fit(train_mat2, tcls)
# predicted_3 = clf_1.predict(test_mat2)
# print(predicted_3[:5])
# with open('predictions_3.dat','w+') as file:
#     for p in predicted_2:
#         file.write(str(p)+"\n")

