from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
from sklearn.externals import joblib


def train_random_forrest(passenger_data):

    x = passenger_data[['PClass', 'Age', 'Sex', 'Family', 'Title']]
    y = passenger_data['Survived']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
                   'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)

    rf_random.fit(x_train, y_train)

    best_n_estimators = rf_random.best_params_['n_estimators']
    best_min_samples_split = rf_random.best_params_['min_samples_split']
    best_min_samples_leaf = rf_random.best_params_['min_samples_leaf']
    best_max_features = rf_random.best_params_['max_features']
    best_max_depth = rf_random.best_params_['max_depth']
    best_bootstrap = rf_random.best_params_['bootstrap']

    rfc = RandomForestClassifier(n_estimators=best_n_estimators, min_samples_split=best_min_samples_split,
                                min_samples_leaf=best_min_samples_leaf, max_features=best_max_features,
                                max_depth=best_max_depth, bootstrap=best_bootstrap)

    rfc.fit(x_train, y_train)

    y_pred = rfc.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    y_pred2 = rfc.predict(x_train)
    print(confusion_matrix(y_train, y_pred2))
    print(classification_report(y_train, y_pred2))
    joblib.dump(rfc, '../models/random_forrest_model.pkl')