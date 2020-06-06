import pandas as pd

#loading up data
df = pd.read_csv('fraud_data.csv')

#Fraud Percentage
print('Fraud Percentage: {}'.format(df['Class'].value_counts()[1]/(df['Class'].value_counts()[0]+df['Class'].value_counts()[1])))

#splitting target and features
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#dividing to train and test subsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Using an SVC model
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.svm import SVC
svc = SVC().fit(X_train, y_train)
y_svc = svc.predict(X_test)
print('Accuracy Score: {}\nRecall Score: {}\nPrecision Score: {}'.format(accuracy_score(y_test, y_svc), recall_score(y_test, y_svc), precision_score(y_test, y_svc)))

#Drawing Confusion Matrix for a modified SVC model with desired threshold
from sklearn.metrics import confusion_matrix
svc_m = SVC(C=1e9, gamma=1e-07).fit(X_train, y_train)
y_decisions = svc_m.decision_function(X_test)
y_decisions[y_decisions>-220] = 1
y_decisions[y_decisions<-220] =0
print(confusion_matrix(y_test, y_decisions))

#fitting a linear regression model and drawing a roc curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import numpy as np
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.decision_function(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred)

#Finding desired points
print('Recall where precision is 0.75: {}'.format(float(recall[np.where(precision==0.75)])))
print('True positive rate where false positive rate is 0.16: {}'.format(tpr[np.argmin(np.abs(fpr-0.16))]))

#Grid search on linear regression model with different parameters
from sklearn.model_selection import GridSearchCV
params = {'C' : [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid_lr = GridSearchCV(lr, param_grid = params, scoring='recall')
y_pred = grid_lr.fit(X_train, y_train).decision_function(X_test)
scores = grid_lr.cv_results_['mean_test_score'].reshape(5,2)
print('Mean test score of grid search on logistic regression using L1 and L2 penalties and different Cs:\n {}' .format(scores))

#Drawing the heatmap
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.heatmap(scores, xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
plt.yticks(rotation=0)
plt.ylabel('C value')
plt.xlabel('Penalty')
