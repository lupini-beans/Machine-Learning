import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE


def problem_b(data):
    pd.crosstab(data.marital, data.y).plot(kind='bar')
    plt.title('Purchase Frequency for Marital Status')
    plt.xlabel('Marital Status')
    plt.ylabel('Frequency of Purchase')

def problem_c(data):
    table=pd.crosstab(data.marital, data.y)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of Marital Status vs Purchase')
    plt.xlabel('Marital Status')
    plt.ylabel('Proportion of Customers')

def problem_d(data):
    cat_vars=['age', 'duration', 'pdays', 'emp_var_rate', 'cons_price_idx']
    for var in cat_vars:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(data[var], prefix=var)
        data1 = data.join(cat_list)
        data = data1

    cat_vars=['age', 'duration', 'pdays', 'emp_var_rate', 'cons_price_idx']
    data_vars = data.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in cat_vars]
    data_final = data[to_keep]
    print("data.columns.values: \n", data.columns.values)

    print('______TO KEEP_____\n', data_final.columns.values)
    
    X = data_final.loc[:, data_final.columns != 'y']
    y = data_final.loc[:, data_final.columns == 'y']
    
    os = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    columns = X_train.columns
    print("X_train: ", X_train)
    print("y_train: ", y_train)

    os_data_X, os_data_y = os.fit_sample(X_train, y_train)
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])

    print("Length of oversampled data is", len(os_data_X))
    print("Number of no subscription in oversampled data", len(os_data_y[os_datay['y']==0]))
    print("Proportion of no subscription data in oversampled data is", len(os_data_y[os_data_y['y']==0])/len(os_data_X))
    print("Proportion of subscription data in oversampled data is", len(os_data_y[os_data_y['y']==1])/len(os_data_X))    


data = pd.read_csv('banking.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))

data['education']=np.where(data['education'] == 'basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] == 'basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] == 'basic.4y', 'Basic', data['education'])

problem_b(data)
#plt.show()

problem_c(data)
#plt.show()

problem_d(data)

