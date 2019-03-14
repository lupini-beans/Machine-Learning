#!/usr/bin/env python
# coding: utf-8

# # Practice 3
# ### Hand-written Digits Recognition using SVM
# Emily Lupini
# 
# ---
# ## Read in the MNIST data set:
# 
# Because the test data on Blackboard did not have labels for the test data I downloaded the dataset from here: http://yann.lecun.com/exdb/mnist/
# 
# Additionally, I modified the load_mnist function from Python Machine Learning Second Edition by Sebestian Raschka, since I was no longer reading in CSV's.

# In[1]:


import os
import struct
import numpy as np


def load_mnist(path='./', kinds=['train', 'test']):
    """Load MNIST data from `path`"""
    
    data = {
        'labels': [],
        'images': []
    }
    
    for kind in kinds:
        labels_path = os.path.join(path, '%s-labels' % kind)
        images_path = os.path.join(path, '%s-images' % kind)
        
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            
            labels = np.fromfile(lbpath, dtype=np.uint8)
            
            data['labels'].append(labels)
            
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII",                                                    imgpath.read(16))
            
            images = np.fromfile(imgpath,                                dtype = np.uint8).reshape(                                len(labels), 784)
            
            images = ((images / 225.) - .5) * 2
            
        
            data['images'].append(images)
            
    return data
    
data = load_mnist()
train_labels, train_images = data['labels'][0], data['images'][0]
test_labels, test_images = data['labels'][1], data['images'][1]

X = train_images
y = train_labels

if len(train_labels) == len(train_images):
    print(len(train_labels), 'training labels read')
else:
    print("Error: {labels} labels read, but {images} images read".format(            labels=len(train_labels), images=len(train_images)))
if len(test_labels) == len(test_images):
    print(len(test_labels), 'testing labels read')
else:
    print("Error: {labels} labels read, but {images} images read".format(            labels=len(test_labels), images=len(test_images)))


# ---
# ## Visualize the Data
# 
# To help render the data I turned, again to the Sebestian Raschka Python Machine Learning book. To start, let's look at each of one of each number, 0-9.

# In[4]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()
for i in range(10):
    img = X[y == i][0].reshape(28,28)
    ax[i].imshow(img, cmap='Greys')
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


# Let's look at the variation in a couple of single digits now.

# In[3]:


def plot_int(num):
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    
    ax = ax.flatten()
    for i in range(25):
        img = train_images[train_labels == num][i].reshape(28,28)
        ax[i].imshow(img, cmap='Greys')
    
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    
plot_int(6)
plot_int(4)
plot_int(9)


# ---
# ## Model 1
# #### SVM
# 
# For the first svm model I'm going to use a linear fit.

# In[6]:


from sklearn.svm import SVC
from sklearn import metrics

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X, y)

y_test_pred = svm.predict(test_images)
ac = metrics.accuracy_score(test_labels, y_test_pred)

print('Linear SVM Accuracy: {ac}%'.format(ac=(ac*100)))
print('Linear SVM Report:\n', metrics.classification_report(test_labels, y_test_pred))


# This is definitely an improvement on the KNN model (below), and I was able to run it on the entire data set in about 10 minutes, unlike KNN which I cancelled at about 30 minutes. Let's take a look at a few images my classifier got wrong.

# In[7]:


def plot_wrong(num):
    wrong_indices = []
    for index, guess in enumerate(y_test_pred):
        if (guess != test_labels[index]) and (test_labels[index] == num):
            wrong_indices.append(index)
    
    # fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    
    print(len(wrong_indices), 'bad predictions found.')
    print(y_test_pred[wrong_indices[0]])
    # ax = ax.flatten()
 
    for i in range(5):
        axes = plt.gca()
        img = test_images[wrong_indices[i]].reshape(28,28)
        axes.imshow(img, cmap='Greys')
        plt.title('Predicted value = {}'.format(y_test_pred[wrong_indices[i]]))
        plt.show()

plot_wrong(3)    


# This is interesting. The 3's that are misclassified as 5's definietly have the characteristic curve of a lower half of a five. The few where it predicted two have a similar curve to what we'd expect for an upper half of a two. The misclassifications are at least understandable. Now I want to try a different kernel, so let's try the Radial Basis Function. 

# In[8]:


svm = SVC(kernel='rbf', C=1.0, random_state=1, gamma='scale')
svm.fit(X, y)

y_test_pred = svm.predict(test_images)
ac = metrics.accuracy_score(test_labels, y_test_pred)

print('Radial Basis Function SVM Accuracy: {ac}%'.format(ac=(ac*100)))
print('Radial Basis Function SVM Report:\n', metrics.classification_report(test_labels, y_test_pred))


# The Radial Basis Function for SVM performed even better than the linear fit. Let's look at the 3's again and see which ones it got wrong.

# In[9]:


plot_wrong(3)


# ---
# ## Model 2
# #### KNN
# 
# Since we just finished up with KNN I was curious what kind of results they would generate. Because Manhattan performed better in the cats, dogs, and pandas project I decided to use that here. The 60,000 image training set is a bit much for KNN, so let's try running it on 1000 images with a test set of 200.

# In[19]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def accuracy_cross_val(dtype, train, test, k):
    axes = plt.gca()
    axes.set_xlim([0, len(train)])
    axes.set_ylim([min(test)-0.2, 1.1])
    
    purple_patch = mpatches.Patch(color='purple', label='Train')
    pink_patch = mpatches.Patch(color='pink', label='Test')
    plt.legend(handles=[purple_patch, pink_patch])
 
    title='Accuracy CrossVal {dtype}: Optimal k: {k}'.format(dtype=dtype, k=k)
    
    axes.plot(list(range(len(test))), test, color='pink', label='Test Scores')
    axes.plot(list(range(len(train))), train, color='purple', label='Train Scores')
    
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.show()

best = {
    'l1': [0, 0, None],
    'l2': [0, 0, None]
}

l1_train_scores = []
l1_test_scores = []
l2_train_scores = []
l2_test_scores = []

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

knn_X_train, knn_X_tt, knn_y_train, knn_y_tt = train_test_split(X, y, test_size=0.9833, random_state=42)
discard1, knn_X_test, discard2, knn_y_test = train_test_split(knn_X_tt, knn_y_tt, test_size=(.004), random_state=42)

print('Test size =', len(knn_X_test))
print('Train size =', len(knn_X_train))


print('###############')
print('## Manhattan ##')
print('###############')

for k in range(1,31):
    model = KNeighborsClassifier(n_neighbors=k, p=1, n_jobs=1)
    model.fit(knn_X_train, knn_y_train)
    
    y_train_pred = model.predict(knn_X_train)
    l1_train_scores.append(metrics.accuracy_score(knn_y_train, y_train_pred))

    y_test_pred = model.predict(knn_X_test)
    ac = metrics.accuracy_score(knn_y_test, y_test_pred)
    l1_test_scores.append(metrics.accuracy_score(knn_y_test, y_test_pred))
    
    if ac > best['l1'][0]:
        best['l1'][0] = ac
        best['l1'][1] = k
        best['l1'][2] = metrics.classification_report(knn_y_test, y_test_pred)
    
    

print('Best Manhattan k = ', best['l1'][1])
print('\tAccuracy = {ac}%'.format(ac=best['l1'][0]*100))
print('\tReport:\n', best['l1'][2])

accuracy_cross_val('L1', l1_train_scores, l1_test_scores, best['l1'][1])

print('###############')
print('## Euclidean ##')
print('###############')

for k in range(1,31):
    model = KNeighborsClassifier(n_neighbors=k, p=2, n_jobs=1)
    model.fit(knn_X_train, knn_y_train)
    
    y_train_pred = model.predict(knn_X_train)
    l2_train_scores.append(metrics.accuracy_score(knn_y_train, y_train_pred))

    y_test_pred = model.predict(knn_X_test)
    ac = metrics.accuracy_score(knn_y_test, y_test_pred)
    l2_test_scores.append(metrics.accuracy_score(knn_y_test, y_test_pred))
    
    if ac > best['l2'][0]:
        best['l2'][0] = ac
        best['l2'][1] = k
        best['l2'][2] = metrics.classification_report(knn_y_test, y_test_pred)
    
    

print('Best Euclidean k = ', best['l2'][1])
print('\tAccuracy = {ac}%'.format(ac=best['l2'][0]*100))
print('\tReport:\n', best['l2'][2])

accuracy_cross_val('L2', l2_train_scores, l2_test_scores, best['l2'][1])


# Surprisingly, there isn't a whole lot of diffrentiation between Euclidean and Manhattan distance powers. Also, the results get progressively worse as our k gets larger. 

# ---
# ## Model 3
# ### Logistic Regression
# 
# So far the Radial Basis fit for SVM has been the most accurate, but I'm curious to see what kind of results a logistic regression model will generate.

# In[10]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='lbfgs', multi_class='auto', n_jobs=-1)

logreg.fit(X, y)

y_test_pred = logreg.predict(test_images)
ac = metrics.accuracy_score(y_test_pred, test_labels)

print('Logistic Regression Accuracy: {ac}%'.format(ac=(ac*100)))
print('Logistic Regression Report:\n', metrics.classification_report(test_labels, y_test_pred))


# Since 8's seem te be the most poorly classified let's see what some of the bad predictions look like.

# In[11]:


plot_wrong(8)


# ---
# ## Model 4
# #### Decision Tree Classifier
# 
# Lastly, let's try out a decision tree classifier.

# In[14]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X, y)

y_test_pred = dtc.predict(test_images)
ac = metrics.accuracy_score(test_labels, y_test_pred)

print('Decision Tree Accuracy: {ac}%'.format(ac=(ac*100)))
print('Decision Tree Report:\n', metrics.classification_report(test_labels, y_test_pred))


# Let's look at some of the 8's again.

# In[15]:


plot_wrong(8)


# Yikes, that didn't go too well.

# ---
# ## Test with the Best
# 
# Since SVM with the Radial Basis Function kernel generated the greatest results, I'd like to see if I can trim the dataset and garner similar results. I'm going to split _only_ the train data into a new test/split ratio. Let's try 70/30 for train/test.

# In[26]:


print('###############')
print('### RBF SVM ###')
print('###############')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('\n\n70/30 on 60,000 items:')
print('\t', len(X_train), 'training values')
print('\t', len(X_test), 'testing values')

svm = SVC(C=1.0, kernel='rbf', random_state=1, gamma='scale')
svm.fit(X_train, y_train)

y_test_pred = svm.predict(X_test)

print('Accuracy = {ac}'.format(ac=metrics.accuracy_score(y_test, y_test_pred)*100))
print('Report:\n', metrics.classification_report(y_test, y_test_pred))


# Still gets pretty solid results, but it also still takes a bit of time. Let's make it even smaller. I'm going to do the same split (70/30), but this time on the test data which only has 10,000 elements.

# In[27]:


X_train, X_test, y_train, y_test = train_test_split(test_images, test_labels, test_size=0.3, random_state=42)

print('\n\n70/30 on 10,000 items:')
print('\t', len(X_train), 'training values')
print('\t', len(X_test), 'testing values')

svm = SVC(C=1.0, kernel='rbf', random_state=1, gamma='scale')
svm.fit(X_train, y_train)

y_test_pred = svm.predict(X_test)

print('Accuracy = {ac}'.format(ac=metrics.accuracy_score(y_test, y_test_pred)*100))
print('Report:\n', metrics.classification_report(y_test, y_test_pred))


# Impressively, this training set is still more accurate than any other training set (except, of course, SVM RBF on the 60,000 training items).
