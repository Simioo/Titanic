from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn import svm
from sklearn.externals import joblib


def train_svm(passenger_data):

    x = passenger_data[['PClass', 'Age', 'Sex', 'Family', 'Title']]
    y = passenger_data['Survived']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    # Designate distributions to sample hyperparameters from
    np.random.seed(123)
    g_range = np.random.uniform(1, 0.5, 20).astype(float)
    c_range = np.random.normal(3, 2, 20).astype(float)

    hyperparameters = {'kernel': ['linear', 'rbf', 'sigmoid'], 'C': c_range, 'gamma': g_range}

    svc = svm.SVC()
    clf = RandomizedSearchCV(svc, param_distributions=hyperparameters, cv=5)
    clf.fit(x_train, y_train)

    best_kernel = clf.best_params_['kernel']
    best_gamma = clf.best_params_['gamma']
    best_c = clf.best_params_['C']

    print(clf.best_params_)

    svc = svm.SVC(kernel = best_kernel, gamma=best_gamma, C=best_c, probability=True)
    svc.fit(x_train, y_train)

    y_pred = svc.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    y_pred2 = svc.predict(x_train)
    print(confusion_matrix(y_train, y_pred2))
    print(classification_report(y_train, y_pred2))
    joblib.dump(svc, '../models/svm_model.pkl')
