import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("dataset.csv")
team = data["Team"]

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

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
X_train_array = np.array(x_train_list)
X_test_array = np.array(x_test_list)

model = LinearDiscriminantAnalysis()
# Train the model using the training sets
model.fit(X_train_array,Y_train_array)
#Predict Output
predicted= model.predict(X_test_array) 
predicted = list(predicted)

# dictionary of names
teams = {0: 'Bayern Munich 2021-22', 1: 'RB Salzburg 2021-22', 2: 'Manchester City 2021-22', 3: 'Sporting CP 2021-22', 4: 'Atlético Madrid 2021-22', 5: 'Manchester Utd 2021-22', 6: 'Benfica 2021-22', 7: 'Ajax 2021-22', 8: 'Villarreal 2021-22', 9: 'Juventus 2021-22', 10: 'Real Madrid 2021-22', 11: 'Paris S-G 2021-22', 12: 'Liverpool 2021-22', 13: 'Inter 2021-22', 14: 'Chelsea 2021-22', 15: 'Lille 2021-22', 16: 'Bayern Munich 2020-21', 17: 'Lazio 2020-21', 18: 'Paris S-G 2020-21', 19: 'Barcelona 2020-21', 20: 'Manchester City 2020-21', 21: "M'Gladbach 2020-21", 22: 'Real Madrid 2020-21', 23: 'Atalanta 2020-21', 24: 'Porto 2020-21', 25: 'Juventus 2020-21', 26: 'Liverpool 2020-21', 27: 'RB Leipzig 2020-21', 28: 'Dortmund 2020-21', 29: 'Sevilla 2020-21', 30: 'Chelsea 2020-21', 31: 'Atlético Madrid 2020-21', 32: 'Bayern Munich 2019-20', 33: 'Chelsea 2019-20', 34: 'Barcelona 2019-20', 35: 'Napoli 2019-20', 36: 'RB Leipzig 2019-20', 37: 'Tottenham 2019-20', 38: 'Manchester City 2019-20', 39: 'Real Madrid 2019-20', 40: 'Atlético Madrid 2019-20', 41: 'Liverpool 2019-20', 42: 'Atalanta 2019-20', 43: 'Valencia 2019-20', 44: 'Paris S-G 2019-20', 45: 'Dortmund 2019-20', 46: 'Lyon 2019-20', 47: 'Juventus 2019-20', 48: 'Liverpool 2018-19', 49: 'Bayern Munich 2018-19', 50: 'Manchester Utd 2018-19', 51: 'Paris S-G 2018-19', 52: 'Ajax 2018-19', 53: 'Real Madrid 2018-19', 54: 'Barcelona 2018-19', 55: 'Lyon 2018-19', 56: 'Tottenham 2018-19', 57: 'Dortmund 2018-19', 58: 'Porto 2018-19', 59: 'Roma 2018-19', 60: 'Manchester City 2018-19', 61: 'Schalke 04 2018-19', 62: 'Juventus 2018-19', 63: 'Atlético Madrid 2018-19', 64: 'Bayern Munich 2017-18', 65: 'Beşiktaş 2017-18', 66: 'Sevilla 2017-18', 67: 'Manchester Utd 2017-18', 68: 'Barcelona 2017-18', 69: 'Chelsea 2017-18', 70: 'Juventus 2017-18', 71: 'Tottenham 2017-18', 72: 'Real Madrid 2017-18', 73: 'Paris S-G 2017-18', 74: 'Liverpool 2017-18', 75: 'Porto 2017-18', 76: 'Manchester City 2017-18', 77: 'Basel 2017-18', 78: 'Roma 2017-18', 79: 'Shakhtar 2017-18'}

# getting names of teams
predictedTeams = []
actualTeams = []
check = []
sameTeam = []

for i in predicted:
    predictedTeams.append(teams[i])

actual = Y_test_array.flatten()
actual = list(actual)
for i in actual:
    actualTeams.append(teams[i])

# checking if it is same team or not
for i in range(len(predicted)):
    if predicted[i] == actual[i]:
        check.append('Same')
    else:
        check.append('Different')

# checking if same team but different year
for i in range(len(actual)):
    team1 = predictedTeams[i].split(' ')[::-1][1:]
    team2 = actualTeams[i].split(' ')[::-1][1:]
    if team1 == team2:
        sameTeam.append("Same Team")
    else:
        sameTeam.append("")
        

final = [predictedTeams, actualTeams, check, sameTeam]
final = pd.DataFrame(final)
final = final.transpose()
final.columns = ["Predicted", "Actual", "Same or Not", "Same Team"]
print(final.to_markdown()) 
