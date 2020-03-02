from sklearn.externals import joblib
from process_dataset import add_columns, get_age_category, avg_diff_for_married_couples, average_master_age, is_master
from process_dataset import married_age, calculate_mean_age, replace_missing_age
import numpy as np


def predict(model, passenger_data, to_predict):

    y = to_predict['Survived'].tolist()

    to_predict = process_input_data(passenger_data, to_predict)
    to_predict = to_predict[['PClass', 'Age', 'Sex', 'Family', 'Title']]

    print(model.predict_proba(to_predict)[:, 1])

    y_predicted = model.predict(to_predict).astype(int)
    print('Accuracy: ', round((np.sum(y == y_predicted) / len(y)) * 100, 2), '%')
    print(y_predicted)


def fill_missing_values(pdata, to_predict):
    """Return modified DataFrame with filled missing values"""

    pdata['Age'] = pdata['Age'].astype(float)
    to_predict['Age'] = to_predict['Age'].astype(float)
    classes = ['1st', '2nd', '3rd']
    sex = ['female', 'male']

    # Filling missing age values for married couples and boys with master title
    average_couple_age_diff = avg_diff_for_married_couples(pdata)
    avg_master_age = average_master_age(pdata)

    for index, row in to_predict.iterrows():

        if is_master(row['Name']) and np.isnan(float(row['Age'])):
            to_predict.at[index, 'Age'] = avg_master_age

        elif np.isnan(float(row['Age'])):

            partner_age = married_age(pdata, row['Name'])

            if partner_age != None:
                if row['Sex'] == 'male':
                    to_predict.at[index, 'Age'] = partner_age + average_couple_age_diff
                elif row['Sex'] == 'female':
                    to_predict.at[index, 'Age'] = partner_age - average_couple_age_diff

    # For all other missing values, fill with average age for each group
    for cls in classes:
        for sx in sex:
            mean = calculate_mean_age(pdata, cls, sx)
            to_predict_modified = replace_missing_age(to_predict, cls, sx, mean)
            to_predict.loc[(to_predict.PClass == cls) & (to_predict.Sex == sx), ['Age']] = to_predict_modified

    return to_predict


def process_input_data(passenger_data, to_predict):

    # Adding title and family column extracted from Name column
    to_predict = add_columns(to_predict)

    # Filling missing Age values
    to_predict = fill_missing_values(passenger_data, to_predict)

    # Replace age values with category numbers 1, 2, 3 and 4
    to_predict.Age = to_predict.Age.apply(get_age_category)

    # Replacing passenger class with numbers 0, 1 and 2
    class_ord_map = {'1st': 1, '2nd': 2, '3rd': 3}
    to_predict["PClass"] = to_predict["PClass"].map(class_ord_map)

    # Replacing sex labels with numbers 0 and 1
    sex_ord_map = {'male': 0, 'female': 1}
    to_predict["Sex"] = to_predict["Sex"].map(sex_ord_map)

    return to_predict


def log_reg_predict(passenger_data, to_predict):
    logreg = joblib.load('../models/log_reg_model.pkl')
    predict(logreg, passenger_data, to_predict)


def svm_predict(passenger_data, to_predict):
    svc = joblib.load('../models/svm_model.pkl')
    predict(svc, passenger_data, to_predict)


def tree_predict(passenger_data, to_predict):
    decision_tree = joblib.load('../models/tree_model.pkl')
    predict(decision_tree, passenger_data, to_predict)


def random_forrest_predict(passenger_data, to_predict):
    random_forrest = joblib.load('../models/random_forrest_model.pkl')
    predict(random_forrest, passenger_data, to_predict)


def bayes_predict(passenger_data, to_predict):
    bayes_predict = joblib.load('../models/bayes_model.pkl')
    predict(bayes_predict, passenger_data, to_predict)


def knn_predict(passenger_data, to_predict):
    knn_predict = joblib.load('../models/knn_model.pkl')
    predict(knn_predict, passenger_data, to_predict)


def voting_predict(passenger_data, to_predict):
    voting_predict = joblib.load('../models/voting_model.pkl')
    predict(voting_predict, passenger_data, to_predict)