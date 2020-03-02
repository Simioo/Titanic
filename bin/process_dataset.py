import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def calculate_mean_age(pdata, pclass, sex):
    """Return mean age for group of passengers specified with function parameters"""
    return pdata[(pdata.PClass == pclass) & (pdata.Sex == sex)].Age.mean(skipna=True)


def replace_missing_age(pdata, pclass, sex, replacement):
    """Replaces age values for group of passengers specified with function parameters with average age for that group"""
    return pdata['Age'][(pdata.PClass == pclass) & (pdata.Sex == sex)].replace(to_replace=[np.nan], value=replacement)


def add_columns(pdata):
    """Add new columns Title and Family and return modified DataFrame

    Family column: 0- no family, 1- has family
    Title column: 1- Special title, 2- Mr/Mrs/Miss/Ms/Master, 3- No title at all
    """
    family_column = []
    title_column = []

    for index, row in pdata.iterrows():

        family_column.append(has_family(pdata, row['Name'], row['PClass']))

        if has_title(row['Name']):
            title_column.append(1)
        elif mr_mrs_title(row['Name']):
            title_column.append(2)
        else:
            title_column.append(3)

    pdata['Family'] = family_column
    pdata['Title'] = title_column
    return pdata


def fill_missing_values(pdata):
    """Return modified DataFrame with filled missing values"""

    pdata['Age'] = pdata['Age'].astype(float)
    classes = ['1st', '2nd', '3rd']
    sex = ['female', 'male']

    # Removing the only passenger with no info about class
    pdata = pdata[pdata.PClass != '*']
    pdata.reset_index()

    # Filling missing age values for married couples and boys with master title
    average_couple_age_diff = avg_diff_for_married_couples(pdata)
    avg_master_age = average_master_age(pdata)

    for index, row in pdata.iterrows():

        if is_master(row['Name']) and np.isnan(float(row['Age'])):
            pdata.at[index, 'Age'] = avg_master_age

        elif np.isnan(float(row['Age'])):

            partner_age = married_age(pdata, row['Name'])

            if partner_age != None:
                if row['Sex'] == 'male':
                    pdata.at[index, 'Age'] = partner_age + average_couple_age_diff
                elif row['Sex'] == 'female':
                    pdata.at[index, 'Age'] = partner_age - average_couple_age_diff

    # For all other missing values, fill with average age for each group
    for cls in classes:
        for sx in sex:
            mean = calculate_mean_age(pdata, cls, sx)
            pdata_modified = replace_missing_age(pdata, cls, sx, mean)
            pdata.loc[(pdata.PClass == cls) & (pdata.Sex == sx), ['Age']] = pdata_modified

    return pdata


def has_title(name):

    titles = ['Lady', 'Colonel', 'Major', 'Sir', 'Madame', 'Countess', 'Dr', 'Captain', 'Col']
    for title in titles:
        if title in name:
            return True

    return False


def mr_mrs_title(name):

    titles = ['Mrs', 'Mr', 'Ms', 'Miss', 'Master']

    for title in titles:
        if title in name:
            return True

    return False


def is_master(name):

    if 'Master' in name:
        return True
    return False


def probably_married(name1, name2):
    """Return boolean value if two people are probably married"""

    titles = ['Lady', 'Colonel', 'Major', 'Master', 'Sir', 'Madame', 'Countess', 'Dr', 'Captain', 'Col', 'Mrs', 'Mr', 'Ms', 'Miss']

    for title in titles:
        name1 = name1.split('(')[0].replace(title, '')
        name2 = name2.split('(')[0].replace(title, '')

    if not ((name1 in name2) or (name2 in name1)):
        return False

    return True


def married_age(pdata, name):

    names = pdata[pdata.Name != name].Name.tolist()
    ages = pdata[pdata.Name != name].Age.tolist()

    for i in range(0, len(names)):

        if names[i] == name:
            continue

        if probably_married(name, names[i]):
            if np.isnan(float(ages[i])):
                return None
            else:
                return float(ages[i])

    return None


def has_family(pdata, name, cls):

    names = pdata[pdata.Name != name].Name.tolist()
    classes = pdata[pdata.Name != name].PClass.tolist()

    for i in range(0, len(names)):
        if (names[i].split(',')[0] == name.split(',')[0]) and (classes[i] == cls):
            return 1

    return 0


def avg_diff_for_married_couples(pdata):
    """Return average age difference between married couples"""
    names = pdata.Name.tolist()
    ages = pdata.Age.tolist()

    diff_sum, num_of_couples = 0, 0

    for i in range(0, len(names)):
        for j in range(i+1, len(names)):

            married = probably_married(names[i], names[j])
            ages_null = np.isnan(float(ages[i])) or np.isnan(float(ages[j]))

            if married and not ages_null:
                diff_sum += abs(float(ages[i]) - float(ages[j]))
                num_of_couples += 1

    return round(diff_sum/num_of_couples, 2)


def get_age_category(age):

    if float(age) < 16:
        return 1
    elif float(age) < 31:
        return 2
    elif float(age) < 61:
        return 3
    else:
        return 4

def average_master_age(pdata):
    """Calculate average age for passenger with Master title"""
    passenger_names = pdata.Name.tolist()
    passenger_ages = pdata.Age.tolist()
    valid_ages = []

    for i in range(0, len(passenger_names)):
        if ('Master' in passenger_names[i]):
            if not np.isnan(float(passenger_ages[i])):
                valid_ages.append(float(passenger_ages[i]))

    return round(sum(valid_ages)/len(valid_ages),2)

def process_dataset():

    passenger_data = pd.read_csv('../data/raw/Titanic_dataset.csv', delimiter=',', names=['Name', 'PClass', 'Age', 'Sex', 'Survived'])

    # Removing first row containing column names
    passenger_data = passenger_data[1:]

    # Adding title and family column extracted from Name column
    passenger_data = add_columns(passenger_data)

    # Filling missing Age values
    passenger_data = fill_missing_values(passenger_data)

    # Replace age values with category numbers 1, 2, 3 and 4
    passenger_data.Age = passenger_data.Age.apply(get_age_category)

    # Replacing passenger class with numbers 0, 1 and 2
    class_ord_map = {'1st': 1, '2nd': 2, '3rd': 3}
    passenger_data["PClass"] = passenger_data["PClass"].map(class_ord_map)

    # Replacing sex labels with numbers 0 and 1
    sex_ord_map = {'male': 0, 'female': 1}
    passenger_data["Sex"] = passenger_data["Sex"].map(sex_ord_map)

    passenger_data.to_csv('../data/processed/processed_dataset.csv', sep=',', index=False)