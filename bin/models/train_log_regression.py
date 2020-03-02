from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import numpy as np
from sklearn.externals import joblib


def train_logistic_regression(passenger_data):

    x = passenger_data[['PClass', 'Age', 'Sex', 'Family', 'Title']]
    y = passenger_data['Survived']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    np.random.seed(123)
    c_range = np.random.normal(1, 0.2, 10).astype(float)
    hyperparameters = {'penalty': ['l1', 'l2'], 'C': c_range}

    clf = RandomizedSearchCV(LogisticRegression(), param_distributions=hyperparameters, cv=5)
    clf.fit(x_train, y_train)

    best_penalty = clf.best_params_['penalty']
    best_c = clf.best_params_['C']

    logreg = LogisticRegression(penalty=best_penalty, C=best_c)
    logreg.fit(x_train, y_train)

    print(metrics.classification_report(y_test, logreg.predict(x_test)))
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
    print('Accuracy of logistic regression classifier on training set: {:.2f}'.format(logreg.score(x_train, y_train)))

    joblib.dump(logreg, '../models/log_reg_model.pkl')
