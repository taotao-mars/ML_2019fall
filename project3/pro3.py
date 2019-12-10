import pandas as pd
import numpy as np

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
           'hours_per_week', 'native_country', 'income']
# using pd to read data
dataset = pd.read_csv('census-income_10percentData.csv', names=columns)
dataset['workclass'] = dataset['workclass'].replace('?', 'Private')
dataset['native_country'] = dataset['native_country'].replace('?', 'United-States')
dataset['occupation'] = dataset['occupation'].replace('?', 'Prof-specialty')
# drop the header
dataset.drop(dataset.index[0], inplace=True)

# Delete duplicate
del dataset['education']
dataset.replace(['<=50K', '>50K'], [-1, 1], inplace=True)
dataset.replace(
    ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India',
     'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam',
     'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
     'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong',
     'Holand-Netherlands'],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    inplace=True)

dataset.replace(
    {'marital_status': {'Married-spouse-absent': 3, 'Married-civ-spouse': 15, 'Married-AF-spouse': 1, 'Divorced': 7,
                        'Never-married': 10, 'Separated': 4,
                        'Widowed': 5
                        },
     'relationship': {'Wife': 2, 'Own-child': 5, 'Husband': 8, 'Not-in-family': 7, 'Other-relative': 1, 'Unmarried': 4
                      },
     'workclass': {'Private': 19, 'Self-emp-not-inc': 12, 'Self-emp-inc': 4, 'Federal-gov': 3, 'Local-gov': 8,
                   'State-gov': 5, 'Without-pay': 1,
                   },
     'occupation': {'Tech-support': 3, 'Craft-repair': 12, 'Other-service': 8, 'Sales': 11, 'Exec-managerial': 10,
                    'Prof-specialty': 13, 'Handlers-cleaners': 6, 'Machine-op-inspct': 7, 'Adm-clerical': 9,
                    'Farming-fishing': 4, 'Transport-moving': 5, 'Priv-house-serv': 1, 'Protective-serv': 2,
                    },
     'race': {'White': 13, 'Asian-Pac-Islander': 5, 'Amer-Indian-Eskimo': 2, 'Other': 1, 'Black': 7
              },
     'sex': {'Female': 1, 'Male': 0
             }
     }, inplace=True)


# split the dataset into 10 folds
def Kfolds(data):
    size = len(data)
    step = size // 10
    folds = [data[i: i + step].sample(frac=1) for i in range(0, size, step)]
    return folds


# split the <=50k and >50k data
morethan50k = dataset[dataset['income'] == 1]
lessthan50k = dataset[dataset['income'] == -1]
# split the more and less into 10 folds respectively
morethan50k = Kfolds(morethan50k)
lessthan50k = Kfolds(lessthan50k)


# make new dataset in 10 folds
def make_new_dataset(morethan50k, lessthan50k):
    new_data = pd.DataFrame(columns=['age', 'workclass', 'fnlwgt', 'education_num', 'marital_status',
                                     'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                                     'hours_per_week', 'native_country', 'income'])
    for i in range(10):
        frames1 = [morethan50k[i], lessthan50k[i]]
        fold = pd.concat(frames1)
        frames2 = [new_data, fold]
        new_data = pd.concat(frames2)
    new_data = Kfolds(new_data)
    return new_data


new_dataset = make_new_dataset(morethan50k, lessthan50k)

# find the information gain
import math


def information_gain(feature, dataset, label):
    feature_num_dict = {}
    for key in feature:
        feature_num_dict[key] = feature_num_dict.get(key, 0) + 1

    feature_more_than_50k_num_dict = {}

    feature_less_than_50k_num_dict = {}
    for key in feature_num_dict.keys():
        for income in dataset[feature.isin([key])].income:
            if income == 1:
                feature_more_than_50k_num_dict[key] = feature_more_than_50k_num_dict.get(key, 0) + 1
            else:
                feature_less_than_50k_num_dict[key] = feature_less_than_50k_num_dict.get(key, 0) + 1

    feature_more_than_50k_p_dict = {}
    feature_less_than_50k_p_dict = {}
    for key in feature_num_dict.keys():
        if key in feature_more_than_50k_num_dict.keys():
            feature_more_than_50k_p_dict[key] = feature_more_than_50k_num_dict[key] / feature_num_dict[key]
        if key in feature_less_than_50k_num_dict.keys():
            feature_less_than_50k_p_dict[key] = feature_less_than_50k_num_dict[key] / feature_num_dict[key]
    weighted_feature_entropy = {}
    for key in feature_num_dict.keys():
        temp1 = 0
        temp2 = 0
        if key in feature_more_than_50k_p_dict.keys():
            temp1 = feature_more_than_50k_p_dict[key] * math.log(feature_more_than_50k_p_dict[key])
        if key in feature_less_than_50k_p_dict.keys():
            temp2 = feature_less_than_50k_p_dict[key] * math.log(feature_less_than_50k_p_dict[key])
        weighted_feature_entropy[key] = -(temp1 + temp2) * feature_num_dict[key] / len(feature)
    p_income = {}
    for key in label:
        p_income[key] = p_income.get(key, 0) + 1
    for key in p_income.keys():
        p_income[key] = p_income[key] / len(feature)

    entropy_income = 0
    for key in p_income.keys():
        temp = - (p_income[key] * math.log(p_income[key], 2))
        entropy_income = entropy_income + temp

    information_gain = 0
    for key in weighted_feature_entropy.keys():
        information_gain = entropy_income - weighted_feature_entropy[key]

    return information_gain


print("__________________________________")
# print('information gain: age: ', information_gain(new_dataset[1].age, new_dataset[1], new_dataset[1].income))
# print('information gain: workclass: ', information_gain(new_dataset[1].workclass, new_dataset[1], new_dataset[1].income))
# print('information gain: fnlwgt: ', information_gain(new_dataset[1].fnlwgt, new_dataset[1], new_dataset[1].income))
# print('information gain: education_num: ', information_gain(new_dataset[1].education_num, new_dataset[1], new_dataset[1].income))
# print('information gain: marital_status: ', information_gain(new_dataset[1].marital_status, new_dataset[1], new_dataset[1].income))
# print('information gain: occupation: ', information_gain(new_dataset[1].occupation, new_dataset[1], new_dataset[1].income))
# print('information gain: relationship: ', information_gain(new_dataset[1].relationship, new_dataset[1], new_dataset[1].income))
# print('information gain: race: ', information_gain(new_dataset[1].race, new_dataset[1], new_dataset[1].income))
# print('information gain: sex: ', information_gain(new_dataset[1].sex, new_dataset[1], new_dataset[1].income))
# print('information gain: capital_gain: ', information_gain(new_dataset[1].capital_gain, new_dataset[1], new_dataset[1].income))
# print('information gain: capital_loss: ', information_gain(new_dataset[1].capital_loss, new_dataset[1], new_dataset[1].income))
# print('information gain: hours_per_week: ', information_gain(new_dataset[1].hours_per_week, new_dataset[1], new_dataset[1].income))
# print('information gain: native_country: ', information_gain(new_dataset[1].native_country, new_dataset[1], new_dataset[1].income))


import matplotlib.pyplot as plt
% matplotlib
inline
import seaborn as sns

plt.style.use('ggplot')
plt.figure(figsize=(800, 400))

income_type = [1, -1]
df = pd.DataFrame({
    'age': pd.to_numeric(new_dataset[1].age.values),
    'occupation': pd.to_numeric(new_dataset[1].relationship.values),
    'income': pd.to_numeric(new_dataset[1].income.values)
})
fg = sns.FacetGrid(data=df, hue='income', hue_order=income_type)
fg.map(plt.scatter, 'age', 'occupation').add_legend()
print("#######income '1' is > 50k, '-1' is <= 50k ########")
plt.show()

new_dataset

# linear Soft SVM
from numpy import *

np.seterr(divide='ignore', invalid='ignore')


# load dataset
def read_data(dataset):
    dataset = pd.DataFrame(dataset)
    col_length = dataset.columns.size
    features = dataset.iloc[:, [0, 5]].apply(pd.to_numeric, errors='ignore')  # 0 is age, 6 is relationship
    label = dataset.iloc[:, -1].apply(pd.to_numeric, errors='ignore').values
    return features, label


def chlipAlpha(a, H, L):
    if a > H:
        a = H
    if L > a:
        a = L
    return a


def calcEk(svm, i):
    matrix = np.multiply(svm.alphas, svm.labelMat).T
    temp = matrix * svm.Ker[:, i] + svm.b
    f = float(temp)
    e = f - float(svm.labelMat[i])
    return e


def kernelTrans(X, data_mat, K_type):
    row, col = np.shape(X)
    Ker = np.mat(np.zeros((row, 1)))
    if K_type[0] == 'linear':  # linear kernel
        Ker = X * data_mat.T
    elif K_type[0] == 'rbf':  # radial bias function
        for i in range(row):
            d = X[i, :] - data_mat
            Ker[i] = d * d.T
        Ker = np.exp(Ker / (-1 * K_type[1] ** 2))
    else:
        raise NameError('Kernel Type not included')
    return Ker


def random_select(a, b):
    num = a
    while (num == a):
        num = int(random.uniform(0, b))
    return num


class data_struct:
    def __init__(self, feature_data, data_type, Cons, stoper, ker_type):
        self.b = 0
        self.s = stoper
        self.feature = feature_data
        self.row = np.shape(feature_data)[0]
        self.labelMat = data_type
        self.Cons = Cons
        self.alphas = np.mat(np.zeros((self.row, 1)))
        self.eCache = np.mat(np.zeros((self.row, 2)))
        self.Ker = np.mat(np.zeros((self.row, self.row)))
        for i in range(self.row):
            temp = kernelTrans(self.feature, self.feature[i, :], ker_type)
            self.Ker[:, i] = temp


def aj_selector(i, svm, Ei):
    k_m = -1
    del_e_m = 0
    e = 0
    svm.eCache[i] = [1, Ei]
    temp = svm.eCache[:, 0].A
    e_list = nonzero(temp)[0]
    e_length = len(e_list)
    if (e_length) > 1:
        for e_k in e_list:
            if e_k == i:
                continue
            Ek = calcEk(svm, e_k)
            deltaE = abs(Ei - Ek)
            if (deltaE > del_e_m):
                k_m = e_k
                del_e_m = deltaE
                e = Ek
        return k_m, e
    else:
        j = random_select(i, svm.row)
        e = calcEk(svm, j)
    return j, e


# update svm data
def e_updator(svm, i):
    e = calcEk(svm, i)
    svm.eCache[i] = [1, e]


def alphas_opt(i, svm):
    e_i = calcEk(svm, i)
    if ((svm.labelMat[i] * e_i < -svm.s) and (svm.alphas[i] < svm.Cons)) or (
            (svm.labelMat[i] * e_i > svm.s) and (svm.alphas[i] > 0)):
        j, Ej = aj_selector(i, svm, e_i)
        al_I_old = svm.alphas[i].copy()
        al_J_old = svm.alphas[j].copy()
        if (svm.labelMat[i] != svm.labelMat[j]):
            var = svm.alphas[j] - svm.alphas[i]
            L = max(0, var)
            temp = svm.Cons + svm.alphas[j] - svm.alphas[i]
            H = min(svm.Cons, temp)
        else:
            temp = svm.alphas[j] + svm.alphas[i] - svm.Cons
            L = max(0, temp)
            H = min(svm.Cons, svm.alphas[j] + svm.alphas[i])
        if L == H:
            return 0
        eta = 2.0 * svm.Ker[i, j] - svm.Ker[i, i] - svm.Ker[j, j]
        if eta >= 0:
            return 0
        svm.alphas[j] -= svm.labelMat[j] * (e_i - Ej) / eta
        svm.alphas[j] = chlipAlpha(svm.alphas[j], H, L)
        e_updator(svm, j)
        if (abs(svm.alphas[j] - al_J_old) < svm.s):
            return 0
        svm.alphas[i] += svm.labelMat[j] * svm.labelMat[i] * (al_J_old - svm.alphas[j])
        e_updator(svm, i)
        b1 = svm.b - e_i - svm.labelMat[i] * (svm.alphas[i] - al_I_old) * svm.Ker[i, i] - svm.labelMat[j] * (
                svm.alphas[j] - al_J_old) * svm.Ker[i, j]
        b2 = svm.b - Ej - svm.labelMat[i] * (svm.alphas[i] - al_I_old) * svm.Ker[i, j] - svm.labelMat[j] * (
                svm.alphas[j] - al_J_old) * svm.Ker[j, j]
        if (0 < svm.alphas[i] < svm.Cons):
            svm.b = b1
        elif (0 < svm.alphas[j] < svm.Cons):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(feature_data, data_type, C, stoper, m_num, ker_type):
    svm = data_struct(np.mat(feature_data), np.mat(data_type).transpose(), C, stoper, ker_type)
    num = 0
    whole = True
    al_change = 0
    while (num < m_num) and ((al_change > 0) or (whole)):
        al_change = 0
        if whole:
            for i in range(svm.row):
                al_change = al_change + alphas_opt(i, svm)
            num = num + 1
        else:
            n_dound = nonzero((svm.alphas.A > 0) * (svm.alphas.A < C))[0]
            for i in n_dound:
                al_change += alphas_opt(i, svm)
            num = num + 1
        if whole:
            whole = False
        elif (al_change == 0):
            whole = True
    return svm


def plot_svm(svm):
    if svm.feature.shape[1] != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    # draw all samples
    for i in range(svm.row):
        if svm.labelMat[i] == -1:
            plt.plot(svm.feature[i, 0], svm.feature[i, 1], 'oy')
        elif svm.labelMat[i] == 1:
            plt.plot(svm.feature[i, 0], svm.feature[i, 1], 'ob')

    non_zero_alphas = np.array(svm.alphas)[np.nonzero(svm.alphas)[0]]
    support_vectors = svm.feature[np.nonzero(svm.alphas)[0]]

    y = np.array(svm.labelMat)[np.nonzero(svm.alphas)].T
    plt.scatter([support_vectors[:, 0]], [support_vectors[:, 1]], s=300, c='r', alpha=0.5, marker='o')


def train_svm(train_dataset, ker_type):
    train_x, train_y = read_data(train_dataset)
    svm = smoP(train_x, train_y, 1, 0.001, 10000, ker_type)
    b = svm.b
    alphas = svm.alphas
    datMat = np.mat(train_x)
    labelMat = np.mat(train_y).transpose()
    svInd = nonzero(alphas)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ker_type)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(train_y[i]):
            errorCount += 1
    return svm


def test_svm(test_dataset, svm, ker_type):
    test_x, test_y = read_data(test_dataset)
    b = svm.b
    alphas = svm.alphas
    datMat = np.mat(test_x)
    labelMat = np.mat(test_y).transpose()
    svInd = nonzero(alphas)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    errorCount_test = 0
    datMat_test = mat(test_x)
    labelMat = mat(test_y).transpose()
    m, n = np.shape(datMat_test)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat_test[i, :], ker_type)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(test_y[i]):
            errorCount_test += 1
    print("the test accuracy rate is: %f" % (1 - (float(errorCount_test) / m)))


def main():
    # 1. train svm in 10-cross-validation
    print("svm with 10-cross-validation")
    print("C=0.001")
    print("____________________")

    test_data = new_dataset[9]
    for i in range(0, 9):
        train_data = new_dataset[i]
        svm_classifier = train_svm(train_data, ('rbf', 1.3))
    print("####### Test #######")
    test_svm(test_data, svm_classifier, ('rbf', 1.3))
    plot_svm(svm_classifier)
    print("\n")


if __name__ == '__main__':
    main()

