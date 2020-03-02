import pandas as pd
from predict import log_reg_predict, svm_predict, tree_predict, random_forrest_predict, knn_predict, voting_predict

passenger_data = pd.read_csv('../data/raw/Titanic_dataset.csv', delimiter=',', names=['Name', 'PClass', 'Age', 'Sex', 'Survived'])

# Removing first row containing column names
passenger_data = passenger_data[1:]

x_predict = pd.read_csv('../data/predict/Titanic_dataset_predict_survival.csv', delimiter=',', names=['Name', 'PClass', 'Age', 'Sex', 'Survived'])

print('Support vector machine prediction: ')
svm_predict(passenger_data, x_predict)

print('Logistic regression prediction: ')
log_reg_predict(passenger_data, x_predict)

print('K nearest neighbor algorithm prediction: ')
knn_predict(passenger_data, x_predict)

print('Voting ensemble prediciton: ')
voting_predict(passenger_data, x_predict)

# Additional models that we try to apply
print('Decision tree prediction: ')
tree_predict(passenger_data, x_predict)

print('Random forrest prediction: ')
random_forrest_predict(passenger_data, x_predict)