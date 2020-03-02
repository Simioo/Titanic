from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier


def knn_train(passenger_data):

    x = passenger_data[['PClass', 'Age', 'Sex', 'Family', 'Title']]
    y = passenger_data['Survived']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    k_range = list(range(1, 31))
    param_grid = dict(n_neighbors=k_range)

    classifier = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, scoring='accuracy')
    classifier.fit(x_train, y_train)

    best_n_neighbors = classifier.best_params_['n_neighbors']
    model = KNeighborsClassifier(n_neighbors=best_n_neighbors)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    joblib.dump(model, '../models/knn_model.pkl')