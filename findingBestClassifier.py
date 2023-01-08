import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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


# Finding best Classifier

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

nochange = ["No Change"]
scaled = ["Scaled"]
pcalist = ["PCA"]
scaledpca = ["Scaled and PCA"]
heading = ["","KNN","SVC","NuSVC","DecisionTree","RandomForest","AdaBoost","GradientBoosting","GaussianNB","LinearDiscriminant","QuadraticDiscriminant"]

# Stats not Changed

for clf in classifiers:
    clf.fit(X_train_array,Y_train_array)
    name = clf.__class__.__name__
    train_predictions = clf.predict(X_test_array)
    acc = accuracy_score(y_test, train_predictions)
    nochange.append(acc * 100)

# Scaled
    
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
X_train_array = np.array(x_train_list)
X_test_array = np.array(x_test_list)

for clf in classifiers:
    clf.fit(X_train_array,Y_train_array)
    name = clf.__class__.__name__
    train_predictions = clf.predict(X_test_array)
    acc = accuracy_score(y_test, train_predictions)
    scaled.append(acc * 100)

# PCA

pca = PCA(n_components=5)
data1 = pca.fit_transform(data)
data = []
for i in data1:
    data.append(list(i))
for i in range(0, len(data)):
    data[i].append(team[i])
data = pd.DataFrame(data)

data.columns = ['a', 'b','c','d','e','team']

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

for clf in classifiers:
    clf.fit(X_train_array,Y_train_array)
    name = clf.__class__.__name__
    train_predictions = clf.predict(X_test_array)
    acc = accuracy_score(y_test, train_predictions)
    pcalist.append(acc * 100)

# SCALED PCA

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
X_train_array = np.array(x_train_list)
X_test_array = np.array(x_test_list)


for clf in classifiers:
    clf.fit(X_train_array,Y_train_array)
    name = clf.__class__.__name__
    train_predictions = clf.predict(X_test_array)
    acc = accuracy_score(y_test, train_predictions)
    scaledpca.append(acc * 100)

# Printing

accuracies = [heading, nochange, scaled, pcalist, scaledpca]
accuracies = pd.DataFrame(accuracies)
print(accuracies.to_string(index=False, header=False))

