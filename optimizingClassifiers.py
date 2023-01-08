import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("dataset.csv")
team = data["Team"]
# getting data

# splitting test and training data
curr = pd.DataFrame(columns=list(data))
train = pd.DataFrame(columns=list(data))
test = pd.DataFrame(columns=list(data))
prev = 0
for i, row in data.iterrows():
    if row[-1] != prev:
        test = pd.concat([test, curr.tail(1)], axis=0)
        train = pd.concat([train, curr.head(len(curr)-1)], axis=0)
        curr = curr.iloc[0:0]

    curr.loc[len(curr.index)] = row
    prev = row[-1]

test = pd.concat([test, curr.tail(1)], axis=0)
train = pd.concat([train, curr.head(len(curr)-1)], axis=0)
curr = curr.iloc[0:0]


x_train = train.iloc[:, : len(list(data)) - 1]
# training data without label

y_train = train.iloc[:,-1:]
# training data label

x_test = test.iloc[:, : len(list(data)) - 1]
# test data without label

y_test = test.iloc[:,-1:]
# test data label

x_train_list = x_train.values.tolist()
y_train_list = y_train.values.tolist()
x_test_list = x_test.values.tolist()
y_test_list = y_test.values.tolist()
# converting to lists

X_train_array = np.array(x_train_list)
Y_train_array = np.array(y_train_list)
X_test_array = np.array(x_test_list)
Y_test_array = np.array(y_test_list)
# numpy arrays

# OPTIMIZING KNN

knn = 1
max = 0
maxknn = 0

while knn < 675:
    model = KNeighborsClassifier(n_neighbors=knn)
    # Train the model using the training sets
    model.fit(X_train_array,Y_train_array)
    #Predict Output
    predicted= model.predict(X_test_array) 
    from sklearn import metrics
    metric = metrics.accuracy_score(y_test, predicted)
    if metric > max:
        max = metric
        maxknn = knn
    knn+= 1

print(maxknn, max)

# OPTIMIZING SVC

kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

best_kernel = None
best_score = -1

for i in kernels:
    model = SVC(kernel=i, C=1)
    model.fit(X_train_array, Y_train_array)
    score = model.score(X_test_array, Y_test_array)
    if score > best_score:
        best_score = score
        best_kernel = i
print('Best Kernel: ', best_kernel) 
