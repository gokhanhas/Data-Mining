# GÖKHAN HAS - 161044067
# CSE 454 - DATA MINING
# ASSIGNMENT 04
# main.py

import math
import pandas as pd
from random import seed
from NaiveBayes import NaiveBayes
from csv import reader
import statsmodels.api as Bayes_Object

def read_csv(filename):
    """
    Parametre olarak gönderilen veri kümesi dosyası açılır ve
    dosya okuma işlemi yapılır.
    :param filename:
    :return:
    """
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def read_data():
    """
    Pandas kütüphanesi kullanılarak veri setinin okunduğu fonksiyondur.
    :return:
    names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'j', 'k', 'l', 'm', 'n' ,'Class']
    """
    names = ['a', 'b', 'c', 'd', 'Class']
    data = pd.read_csv('SampleData.csv', names=names)
    return data


def calculate_f1Score(listResult):
    """
    F1 değerinin hesaplandığı fonksiyondur.
    :param listResult:
    :return:
    """
    val = 0
    for i in listResult:
        val += i
    return val / len(listResult)


def mean(list_x):
    """
    Listenin ortalamasını bulur.
    :param list_x:
    :return:
    """
    return float(sum(list_x)) / len(list_x)

def biderectional_wrapper_method(train, result):
    """
    Wrapper bidecretional method, hem forward_selection hem de backward_selection kullanılır.
    Önceden eklenmiş özelliklerin önemini de kontrol eder ve önceden seçilmiş özelliklerden herhangi birini önemsiz bulursa,
    geriye doğru eleme yoluyla o belirli özelliği kaldırır.
    Model olarak statmodels kütüphanesi kullanılmıştır.
    :param train:
    :param result:
    :param SL_in:
    :param SL_out:
    :return:
    """
    clear_dataset = train.columns.tolist()
    get_features = []
    # Forward_selection ile başlanır.
    while len(clear_dataset) > 0:
        features = list(set(clear_dataset) - set(get_features))
        new_features = pd.Series(index=features, dtype='float64')
        for new_column in features:
            model = Bayes_Object.OLS(result, Bayes_Object.add_constant(train[get_features+[new_column]])).fit()
            new_features[new_column] = model.pvalues[new_column]
        min_p_value = new_features.min()
        # Modelde anlamlılık seviyesi 0.04 seçilmiştir.
        # Backward_selection ile devam edilir.
        if min_p_value < 0.04:
            get_features.append(new_features.idxmin())
            while len(get_features) > 0:
                best_features_with_constant = Bayes_Object.add_constant(train[get_features])
                # Feature lar arasında anlamsız olanlar çıkarılmalıdır.
                p_values = Bayes_Object.OLS(result, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if max_p_value >= 0.04:
                    excluded_feature = p_values.idxmax()
                    get_features.remove(excluded_feature)
                else:
                    break
        else:
            break
    return get_features


def result(train, result):
    """
    Pearson correlation coefficient formülü uygulanır.
    :param train:
    :param result:
    :return:
    """
    size = len(train)
    # Girdilerin ortalaması alınır.
    train_mean = mean(train)
    result_mean = mean(result)


    top_part, x, y = 0, 0, 0
    for index in range(size):
        # Değerlerin ortalamalarına göre uzaklıklarının bulunması
        xi = train[index] - train_mean
        yi = result[index] - result_mean
        # PCC formülü uygulanması
        top_part += xi * yi
        # Değerlerin karelerinin alınması
        x += xi * xi
        y += yi * yi

    return top_part / math.sqrt(x * y)


def main():
    seed(1)
    #################################################
    ############# NAIVE BAYES METHOD ################
    #################################################

    BayesObject = NaiveBayes(n=5)
    data = read_csv("SampleData.csv")

    # VERİ KÜMESİ TEMİZLEME İŞLEMİ ...
    for i in range(len(data[0]) - 1):
        for values in data:
            values[i] = float(values[i].strip())
    # Her bir satırda bulunan değerler ayırıcıya göre ayrılır.
    class_values = [values[len(data[0]) - 1] for values in data]
    # Her bir satırdaki verilerin hangi sınıfa ait olduğu anlaşılır.
    # Ait oldukları sınıf bilgisi dosyanın son sütünunda bulunur.
    class_values_set = set(class_values)
    control = dict()
    for i, value in enumerate(class_values_set):
        control[value] = i
    for row in data:
        # Hangi sınıfa ait olduğu satırdaki son parametreden çekilir.
        row[len(data[0]) - 1] = control[row[len(data[0]) - 1]]


    f1_values = BayesObject.naive_bayes(data, n_folds=5)
    print('NAIVE BAYES METHOD, F1 SCORE : ', calculate_f1Score(f1_values))


    #################################################
    ########## FILTER FEATURE SELECTION #############
    #################################################
    print("\n----------------------------------------------------------------------------\n")

    ffs_data = read_data()
    ffs_class_label = ffs_data['Class']
    pearson = []

    for column in ffs_data:
        if column != 'Class':
            # Veri kümesindeki her bir satır için PCC değerlerinin hesaplanması
            data1 = ffs_data[column]
            pearson.append(result(data1, ffs_class_label))

    print("PCC : ", pearson)
    for i in range(len(data)):
        deletedCount = 0
        for j in range(len(pearson)):
            if (pearson[j] < 0.40):
                # PCC değerleri altında olan satırların yani özelliklerin çıkarılarak NAIVE BAYES uygulanması
                data[i].remove(data[i][j - deletedCount])
                deletedCount += 1

    f1_values = BayesObject.naive_bayes(data, n_folds=5)
    print('FILTER FEATURE SELECTION, F1 SCORE : ', calculate_f1Score(f1_values))
    print("\n----------------------------------------------------------------------------\n")

    #################################################
    ################ WRAPPER METHODS ################
    #################################################
    wrapper_method_data = read_data()
    train_dataset = wrapper_method_data.drop("Class", 1)
    result_dataset = wrapper_method_data['Class']
    biderectional_wrapper_method(train_dataset, result_dataset)
    BayesObject = NaiveBayes(n=5)
    data = read_csv("SampleData.csv")
    # VERİ KÜMESİ TEMİZLEME İŞLEMİ ...
    for i in range(len(data[0]) - 1):
        for values in data:
            values[i] = float(values[i].strip())
    # Her bir satırda bulunan değerler ayırıcıya göre ayrılır.
    class_values = [values[len(data[0]) - 1] for values in data]
    # Her bir satırdaki verilerin hangi sınıfa ait olduğu anlaşılır.
    # Ait oldukları sınıf bilgisi dosyanın son sütünunda bulunur.
    class_values_set = set(class_values)
    control = dict()
    for i, value in enumerate(class_values_set):
        control[value] = i
    for row in data:
        # Hangi sınıfa ait olduğu satırdaki son parametreden çekilir.
        row[len(data[0]) - 1] = control[row[len(data[0]) - 1]]
    f1_values = BayesObject.naive_bayes(data, n_folds=5)
    print('WRAPPER METHOD, F1 SCORE : ', calculate_f1Score(f1_values))

    print("\n----------------------------------------------------------------------------\n")

    #################################################
    ###################### PCA ######################
    #################################################
    BayesObject_pca = NaiveBayes(n=1)
    data = read_csv("SampleData.csv")

    # VERİ KÜMESİ TEMİZLEME İŞLEMİ ...
    for i in range(len(data[0]) - 1):
        for values in data:
            values[i] = float(values[i].strip())
    # Her bir satırda bulunan değerler ayırıcıya göre ayrılır.
    class_values = [values[len(data[0]) - 1] for values in data]
    # Her bir satırdaki verilerin hangi sınıfa ait olduğu anlaşılır.
    # Ait oldukları sınıf bilgisi dosyanın son sütünunda bulunur.
    class_values_set = set(class_values)
    control = dict()
    for i, value in enumerate(class_values_set):
        control[value] = i
    for row in data:
        # Hangi sınıfa ait olduğu satırdaki son parametreden çekilir.
        row[len(data[0]) - 1] = control[row[len(data[0]) - 1]]
    f1_values = BayesObject_pca.naive_bayes(data, 5)
    print('PCA TOOL, F1 SCORE : ', calculate_f1Score(f1_values))
    print("\n----------------------------------------------------------------------------\n")

    #################################################
    ###################### LDA ######################
    #################################################
    BayesObject_lda = NaiveBayes(n=2)
    data = read_csv("SampleData.csv")
    # VERİ KÜMESİ TEMİZLEME İŞLEMİ ...
    for i in range(len(data[0]) - 1):
        for values in data:
            values[i] = float(values[i].strip())
    # Her bir satırda bulunan değerler ayırıcıya göre ayrılır.
    class_values = [values[len(data[0]) - 1] for values in data]
    # Her bir satırdaki verilerin hangi sınıfa ait olduğu anlaşılır.
    # Ait oldukları sınıf bilgisi dosyanın son sütünunda bulunur.
    class_values_set = set(class_values)
    control = dict()
    for i, value in enumerate(class_values_set):
        control[value] = i
    for row in data:
        # Hangi sınıfa ait olduğu satırdaki son parametreden çekilir.
        row[len(data[0]) - 1] = control[row[len(data[0]) - 1]]

    f1_values = BayesObject_lda.naive_bayes(data, 5)
    print('LDA TOOL, F1 SCORE : ', calculate_f1Score(f1_values))


if __name__ == '__main__':
    main()