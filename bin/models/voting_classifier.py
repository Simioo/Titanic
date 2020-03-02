from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics


def voting_classifier_train(passenger_data):

    x = passenger_data[['PClass', 'Age', 'Sex', 'Family', 'Title']]
    y = passenger_data['Survived']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

    logreg = joblib.load('../models/log_reg_model.pkl')
    svc = joblib.load('../models/svm_model.pkl')
    knn = joblib.load('../models/knn_model.pkl')
    random_forrest = joblib.load('../models/random_forrest_model.pkl')

    voting_classifier = VotingClassifier(estimators=[('lr', logreg), ('svc', svc), ('knn', knn),
                                                     ('rf', random_forrest)], voting='soft')
    voting_classifier.fit(x_train, y_train)

    print('Accuracy on training set: ', round(accuracy_score(y_train, voting_classifier.predict(x_train)) * 100, 2), '%')
    print('Accuracy on test set: ', round(accuracy_score(y_test, voting_classifier.predict(x_test)) * 100, 2), '%')
    print(metrics.classification_report(y_test, logreg.predict(x_test)))
    joblib.dump(voting_classifier, '../models/voting_model.pkl')