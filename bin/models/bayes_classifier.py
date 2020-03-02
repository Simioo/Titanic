from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import metrics


def train_bayes_classifier(passenger_data):

    x = passenger_data[['PClass', 'Age', 'Sex', 'Family', 'Title']]
    y = passenger_data['Survived']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    model = GaussianNB()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    joblib.dump(model, '../models/bayes_model.pkl')

