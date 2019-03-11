import pandas as pd

#######
## a ##
#######

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("pima-indians-diabetes-database.csv", header=None, names=col_names)

feature_col = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'] 
X = pima[feature_col]
y = pima.label

#######
## b ##
#######
# Load the data and split the data to be 70% training and 30% testing randomly.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#######
## c ##
#######
# Use the 70% training data to train you model (by using LogisticRegression within sklearn.linear_model)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

#######
## d ##
#######
#Test your model by the testing data and print out confusion matrix, accuracy, precision and recall rate, plot ROC curve and AUC value in the figure.
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

#Heatmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, fmt='g')
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('Confusion martix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

plt.show()

print('\n\n\n\n_______________________________________________________')
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print('_______________________________________________________\n\n\n\n')

# ROC Curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="Data 1, AUC = "+str(auc))
plt.legend(loc=4)
plt.show()

