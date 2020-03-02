from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.metrics import classification_report


def train_decision_tree(passenger_data):

    x = passenger_data[['PClass', 'Age', 'Sex', 'Family', 'Title']]
    y = passenger_data['Survived']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)

    print('Accuracy on training set: ', round(accuracy_score(y_train, clf.predict(x_train)) * 100, 2), '%')
    print('Accuracy on test set: ', round(accuracy_score(y_test, y_pred) * 100, 2), '%')
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, '../models/tree_model.pkl')