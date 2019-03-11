import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

animals = ['cats', 'dogs', 'panda']

def preprocess(base_path, animals=animals):
    data = []
    labels = []
    for animal in animals:

        # Bug with os that doesn't exclude hidden files: 
            # pathlib allows selection to prevent this bug
        animal_dir = Path(base_path) / animal
        for image_name in animal_dir.glob("*.jpg"):
            image = cv2.imread(str(image_name))
            image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
            labels.append(animal)
            data.append(image)
    
    # Vectorize the images
    mydata = np.array(data)
    mydata = mydata.reshape((mydata.shape[0],3072))

    # Label Encoding
    le=preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    mylabels = np.array(labels)

    return (mydata, mylabels, le)


def accuracy_cross_val(type, train, validate, best):
    axes = plt.gca()
    axes.set_xlim([0, len(train)])
    axes.set_ylim([min(validate) -.2, 1.1])

    purple_patch = mpatches.Patch(color='purple', label='Train')
    pink_patch = mpatches.Patch(color='pink', label='Validate')
    plt.legend(handles=[purple_patch, pink_patch])

    title="Accuracy CrossVal {type}: Optimal k: {k}".format(type=type, k=best['k'])

    axes.plot(list(range(1,31)), validate, color='pink', label='Validate Scores')
    axes.plot(list(range(1,31)), train, color='purple', label='Train Scores')
    plt.title(title)

    plt.show()


# Read in the images
print('__________ Begin Reading Images __________')
data, labels, le = preprocess("KNN/animals")
print(labels)
print('__________ Success, Images Read __________')

# Assign X and y
X = data
y = labels

# Train, test, and split
print('\n__________ Begin Train Test Split __________')
X_train, X_tv, y_train, y_tv = train_test_split(X, y, test_size=0.30, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_tv, y_tv, test_size=0.33, random_state=42)
print(y_valid)
print('_____ Train Test Validate Split Done _______') 


# Best fit storage
best_manhattan = {
    'accuracy': 0,
    'k': 0,
    'report': 'none'
}

best_euclidean = {
    'accuracy': 0,
    'k': 0,
    'report': 'none'
}

# Cross Validation Accuracy for Validate and Train
l1_train_scores = []
l1_validate_scores = []
l2_train_scores = []
l2_validate_scores = []

# Model: Manhattan Distance
print('\n\n______________________ Validate ________________________')

print('\n########################')
print('#### L1 | Manhattan ####')
print('########################')
for k in range(1, 31):

    model = KNeighborsClassifier(n_neighbors=k, p=1, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    l1_train_scores.append(metrics.accuracy_score(y_train, y_train_pred))
    
    y_valid_pred = model.predict(X_valid)
    ac = metrics.accuracy_score(y_valid, y_valid_pred)
    l1_validate_scores.append(ac)

    if ac > best_manhattan['accuracy']:
        best_manhattan['accuracy'] = ac
        best_manhattan['k'] = k
        best_manhattan['report'] = metrics.classification_report(y_valid, y_valid_pred, target_names=le.classes_)

print('\nDistance Type: L1 | Manhattan')
print('\tBest k =', best_manhattan['k'], '\n\tAccuracy = {mostaccurate}%'.format(mostaccurate = best_manhattan['accuracy'] * 100))
print('\tBest report =\n', best_manhattan['report'])

accuracy_cross_val('L1', l1_train_scores, l1_validate_scores, best_manhattan)

print('\n\n######################')
print('### L2 | Euclidean ###')
print('######################')
for k in range(1, 31):

    model = KNeighborsClassifier(n_neighbors=k, p=2, n_jobs=-1)
    model.fit(X_train,y_train)
    
    y_train_pred = model.predict(X_train)
    l2_train_scores.append(metrics.accuracy_score(y_train, y_train_pred))
    
    y_valid_pred = model.predict(X_valid)
    ac = metrics.accuracy_score(y_valid, y_valid_pred)
    l2_validate_scores.append(ac)

    if ac > best_euclidean['accuracy']:
        best_euclidean['accuracy'] = ac
        best_euclidean['k'] = k
        best_euclidean['report'] = metrics.classification_report(y_valid, y_valid_pred, target_names=le.classes_)

print('\nDistance Type: L2 | Euclidean')
print('\tBest k =', best_euclidean['k'], '\n\tAccuracy: {mostaccurate}%'.format(mostaccurate = best_euclidean['accuracy'] * 100))
print('\tBest report =\n', best_euclidean['report'])

# CrossVal Accuracy for L2
accuracy_cross_val('L2', l2_train_scores, l2_validate_scores, best_euclidean)

print('\n\n______________________ Results ________________________')
if best_manhattan['accuracy'] > best_euclidean['accuracy']:
    print('\n\nTest Results: ')
    model = KNeighborsClassifier(n_neighbors=best_manhattan['k'], p=1, n_jobs=-1)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    report = metrics.classification_report(y_test, y_test_pred, target_names=le.classes_)
    print('Best Distance: L1 | Manhattan')
    print('Best k:', best_manhattan['k'])
    print('Test accuracy:', metrics.accuracy_score(y_test, y_test_pred))
    print('Test Validation Report: \n', report)


else:
    print('\n\nTest Results: ')
    model = KNeighborsClassifier(n_neighbors=best_euclidean['k'], p=2, n_jobs=-1)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    report = metrics.classification_report(y_test, y_test_pred, target_names=le.classes_)
    print('Best Distance: L2 | Euclidean')
    print('Test k:', best_euclidean['k'])
    print('Test accuracy:', metrics.accuracy_score(y_test, y_test_pred))
    print('Test Validation Report: \n', report)
