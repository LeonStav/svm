#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Ολές οι βιβλιοθήκες που χρησιμοποιήσαμε στον παρακάτω κώδικα
from scipy.stats import mode
import numpy as np
from time import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import random
from IPython.display import display, HTML
from itertools import chain
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
import warnings
warnings.filterwarnings('ignore')
import keras
import tensorflow as tf
print(tf.__version__) #ελέγχω για να δώ εάν έχω την σωστή έκδοση του tensorflow για να φορτώσω το αρχείο mnist


# In[3]:


#Φορτώνουμε το αρχείο MNIST απο τα dataset του keras tensorflow
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


# In[4]:


#Μετράμε το συνολικό ανα αριθμό και τα οπτικοποιούμε μέσω seaborn
sb.countplot(train_labels)
plt.show()


# In[7]:


#Μειώνω την διάσταση των numpyarray απο 3 σε 2, ουσιαστικά πολλαπλασιάζω τα pixel 28*28=784
nsamples, nx, ny = train_images.shape
train_images = train_images.reshape((nsamples,nx*ny))
train_images.shape


# In[8]:


#Μειώνω την διάσταση των numpyarray απο 3 σε 2, ουσιαστικά πολλαπλασιάζω τα pixel 28*28=784
msamples, mx, my = test_images.shape
test_images = test_images.reshape((msamples,mx*my))
test_images.shape


# In[52]:


#Θα χρησιμοποιήσω Support Vector Classifier για να δώ πώς προσαρμόζονται τα δεδομένα σε γραμμικά μοντέλα
svm = LinearSVC(dual=False)
svm.fit(train_images, train_labels)


# In[27]:


svm.coef_ #Τα βάρη που έχουν καταχωρηθεί στο χαρακτηριστικά με γραμμικο kernel
svm.intercept_ #Σταθερές της συνάρτησης απόφασης


# In[110]:


# Η ακρίβεια των δεδομένων στο γραμμικό μοντέλο
accuracy_score(test_labels, labels_pred)


# In[54]:


# Χρησιμοποιώ heatmap για να δώ ποιοί συνδιασμοί δεδομένων έχουν περισσότερα misclassifications  
cm = confusion_matrix(test_labels, labels_pred)
plt.subplots(figsize=(10, 6))
sb.heatmap(cm, annot = True, fmt = 'g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Heatmap Matrix")
plt.show()


# In[9]:


#Linear SVC για πολλαπλούς παράγοντες κόστους C
acc = []
acc_tr = []
coefficient = []
for c in [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]:
    svm = LinearSVC(dual=False, C=c)
    svm.fit(train_images, train_labels)
    coef = svm.coef_
    
    p_tr = svm.predict(train_images)
    a_tr = accuracy_score(train_labels, p_tr)
    
    pred = svm.predict(test_images)
    a = accuracy_score(test_labels, pred)
    
    coefficient.append(coef)
    acc_tr.append(a_tr)
    acc.append(a)


# In[10]:


#Σχεδιάγραμμα κόστους και ακρίβειας του μοντέλου
c = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

plt.subplots(figsize=(10, 5))
plt.semilogx(c, acc,'-gD' ,color='red' , label="Testing Accuracy")
plt.semilogx(c, acc_tr,'-gD' , label="Training Accuracy")
#matplot.xticks(L,L)
plt.grid(True)
plt.xlabel("Cost Parameter C")
plt.ylabel("Accuracy")
plt.legend()
plt.title('Accuracy versus the Cost Parameter C (log-scale)')
plt.show()


# In[45]:


#Eπιλέγω το μοντέλο με την μεγαλύτερη ακρίβεια στα testing data
svm_coef = coefficient[0]
svm_coef.shape


# In[13]:


#Support Vector Classification με RBF kernel
#SVC για διάφορους παραγόντες κόστους C και Gamma
coefficient = []
n_supp = []
sup_vec = []
i = 0
df1 = pd.DataFrame(columns = ['c','gamma','train_acc','test_acc'])
for c in [0.01, 0.1, 1, 10, 100]:
    for g in [0.01, 0.1, 1, 10, 100]:
        svm = SVC(kernel='rbf', C=c, gamma=g)
        model = svm.fit(test_images, test_labels)
        globals()['model%s' % i] = model
        d_coef = svm.dual_coef_
        support = svm.n_support_
        sv = svm.support_
    
        p_tr = svm.predict(test_images)
        a_tr = accuracy_score(test_labels, p_tr)
    
        pred = svm.predict(test_images)
        a = accuracy_score(test_labels, pred)
    
        coefficient.append(d_coef)
        n_supp.append(support)
        sup_vec.append(sv)
        df.loc[i] = [c,g,a_tr,a]
        i=i+1


# In[14]:


#Dataframe με τις τιμές ακρίβειας στο train data και test data
df1


# In[15]:


pd.DataFrame(coefficient[15]) # dual_coef_


# In[30]:


pd.DataFrame(n_supp[15]) # n_support_


# In[21]:


#Support Vector Classification με πολυωνυμικό kernel
coefficient = []
n_supp = []
sup_vec = []
i = 0
df2 = pd.DataFrame(columns = ['c','degree','train_acc','test_acc'])
for c in [0.01, 0.1, 1, 10, 100]:
    for d in [2,3,4,5,6]:
        svm = SVC(kernel='poly', C=c, degree=d)
        model = svm.fit(test_images, test_labels)
        globals()['model%s' % i] = model
        d_coef = svm.dual_coef_
        support = svm.n_support_
        sv = svm.support_
    
        p_tr = svm.predict(test_images)
        a_tr = accuracy_score(test_labels, p_tr)
    
        pred = svm.predict(test_images)
        a = accuracy_score(test_labels, pred)
    
        coefficient.append(d_coef)
        n_supp.append(support)
        sup_vec.append(sv)
        df.loc[i] = [c,d,a_tr,a]
        i=i+1


# In[22]:


df2


# In[23]:


pd.DataFrame(coefficient[20]) # dual_coef_


# In[24]:


pd.DataFrame(n_supp[20]) # n_support_


# In[48]:


#Χρησιμοποιούμε Standardscaler για τις μεταβλητές και έχουμε επιλέξει το πολυωνυμικό για λόγους ακρίβειας
step = [('scaler', StandardScaler()), ('SVM', SVC(kernel='poly'))]
pipeline = Pipeline(step) # define Pipeline object


# In[49]:


#Χρησιμοποιούμε το GridSearchCV για να αποφασίσουμε τις τιμές των gamma και C
parameters = {'SVM__C':[0.001, 0.1, 100, 10e5], 'SVM__gamma':[10,1,0.1,0.01]}
grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)


# In[51]:


#Δοκιμάζουμε το μοντέλο και προσθέτουμε τις παραμέτρους που βρήκαμε
grid.fit(train_images, train_labels)
print ("score = %3.2f" %(grid.score(test_images, test_labels)))
print ("best parameters from train data: ", grid.best_params_)


# In[53]:


#H προβλέψεις των ετικετών στο test
labels_pred = grid.predict(test_images)


# In[19]:


#Οι προβλεπόμενες 5
print (labels_pred[100:105])


# In[20]:


#Οι τιμές των ίδιων θέσεων στο αρχείο test
print (test_labels[100:105])


# In[83]:


#Eδώ τεστάρουμε τις παραμέτρους και παρουσιάζουμε εάν είναι μονά ή ζυγά τα στοιχεία που βρήκαμε
for i in (np.random.randint(0,10000,1)):
 two_d = (np.reshape(test_images[i], (28, 28)) * 255).astype(np.uint8)
 plt.title('predicted label: {0}'. format(labels_pred[i]))
 plt.imshow(two_d, interpolation='nearest', cmap='gray')
 plt.show()
 if labels_pred[i] == [1]:
    print('it is an odd number')
 elif labels_pred[i] == [3]:
    print('it is an odd number')
 elif labels_pred[i] == [5]:
    print('it is an odd number')
 elif labels_pred[i] == [7]:
    print('it is an odd number')
 elif labels_pred[i] == [9]:
    print('it is an odd number')
 else:
    print('it is an even number')

