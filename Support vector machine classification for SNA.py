"""izadi
"""
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.getcwd()
os.chdir(r'D:\desktop\Python_DM_ML_BA\ML\SVM')
df = pd.read_csv('Social_Network_Ads.csv')
df.shape
df.head()
X = df.drop(['User ID', 'Gender','Purchased'],axis=1).values
y = df.Purchased.values

plt.scatter(df.Age, df.EstimatedSalary)
plt.show()
#import Normalisation package
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled =  sc.fit_transform(X)
#split train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,random_state=0)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,roc_curve, auc ,confusion_matrix
model = SVC(kernel='poly', degree=3)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_pred, y_test)
confusion_matrix(y_pred, y_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
roc_auc

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('white', 'black'))(i), label = j)
    plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


