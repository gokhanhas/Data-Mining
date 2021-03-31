# GÖKHAN HAS - 161044067
# CSE 454 - DATA MINING
# ASSIGNMENT 04
# NaiveBayes.py

from random import randrange, seed
from math import sqrt
from math import exp
from math import pi
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import helper


seed(1)

class NaiveBayes:
    def __init__(self, n=0):
        self.n = n
        ## Alttaki işlemler LDA VE PCA hesaplanmasında kullanılacak olan araçlar içindir.
        self.database = self.read_data()
        self.X = self.database.drop('Class', 1)
        self.Y = self.database['Class']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=0, shuffle=False)

    def read_data(self):
        """
        Pandas kütüphanesi kullanılarak veri setinin okunduğu fonksiyondur.
        :return:
        names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'j', 'k', 'l', 'm', 'n', 'Class']
        """
        names = ['a', 'b', 'c', 'd', 'Class']
        data = pd.read_csv('SampleData.csv', names=names)
        return data

    def apply_bayes(self, train_data, test_data):
        """
        Train_data parametresiyle alınan veri kümesi değerleri sınıflarına göre
        ayrılır. Daha sonra her sınıfa ait veriler için gerekli matematiksel hesaplamalar
        yapılır.
        :param train_data:
        :param test_data:
        :return:
        """
        data_values_and_class = dict()
        for i in range(len(train_data)):
            train_one_value = train_data[i]
            class_value = train_one_value[-1]
            # Eğer o sınıfa ait değilse boş atama işlemi yapılırç
            if class_value not in data_values_and_class:
                data_values_and_class[class_value] = []
            # Eğer değer o sınıfa ait ise ait olduğu sınıfının listesine eklenir.
            data_values_and_class[class_value].append(train_one_value)

        # Train_data kaç tane sınıfa ait olduğuna, ve bu değere bölünerek bir dict
        # veri yapısı oluşturulur.
        class_and_values = dict()
        for class_value, value in data_values_and_class.items():
            class_and_values[class_value] = self.calculate_math_metrics(value)

        # Test veri kümesinde bulunan değerlerin tahmin edilerek döndürülmesi
        predicted_values = []
        for value in test_data:
            result = self.find_best(class_and_values, value)
            if self.n >= 3:
                print("Okunan Satır : ", value  , "  Tahmin Edilen : ",result)
            predicted_values.append(result)
        return predicted_values

    def calculate_math_metrics(self, dataset):
        """
        Her bir satırda bulunan verilerin matematiksel olarak hesaplanması yapılır.
        :param dataset:
        :return:
        """
        values = [(self.mean(column), self.standart_derivation(column), len(column)) for column in zip(*dataset)]
        # Hesaplanan değerin bulundu class, veri kümesinde sonda olduğu için ait olduğu class silinmelidir.
        del (values[-1])
        return values

    def mean(self, numbers):
        """
        Satırın ortalamasını hesaplayan fonksiyondur.
        :param numbers:
        :return:
        """
        return sum(numbers) / float(len(numbers))

    # Calculate the standard deviation of a list of numbers
    def standart_derivation(self, value_list):
        """
        Parametre olarak gönderilen listede bulunan elemanların ortalamasını alarak, standart sapma için gerekli olan
        geometrik çarpımı uygular. Formulü uyguladıktan sonra değeri geri döndürür.
        :param numbers:
        :return:
        """
        mean_values = self.mean(value_list)
        variance = sum([(x - mean_values) ** 2 for x in value_list]) / float(len(value_list))
        return sqrt(variance)

    def find_best(self, summaries, row):
        """
        Her bir satır için hangi sınıfa ait olduğu tahmin edilir.
        :param summaries:
        :param row:
        :return:
        """
        pro_values = self.get_predict(summaries, row)
        best_predict, best_value = None, -1
        for class_value, prob in pro_values.items():
            if best_predict is None or prob > best_value:
                best_value = prob
                # Tahmin edilen en iyi değerin bulunması gerekmektedir.
                best_predict = class_value
        return best_predict

    def get_predict(self, train_value, data_value):
        """
        Veri kümesinde bulunan değerlerin hangi sınıfa ait olduğunu bulan fonksiyonudr.
        Veri kümesinin tüm elemanları üzerinde gezinerek, ortalama, standart sapma ve gauss olasılık
        dağılımını kullanarak verinin hangi sınıfa ait olduğu tahmin edilir.
        :param train_value:
        :param data_value:
        :return:
        """
        """
        get_dataset_values = sum([train_value[label][0][2] for label in train_value])
        results = dict()
        # Veri kümesinin değerlerini ve hangi sınıfa ait olduğu bilgisi alınır.
        for dataset_one_value, dataset_which_class in train_value.items():
            results[dataset_one_value] = train_value[dataset_one_value][0][2] / float(get_dataset_values)
            # Her bir satırın ortalama ve standart sapması bulunarak gauss olasılık dağılımı hesaplanır. Bu değer
            # sayının kendisiyle çarpılarak bir tahmin sözlüğe eklenir.
            for i in range(len(dataset_which_class)):
                mean, standart_dev, _ = dataset_which_class[i]
                results[dataset_one_value] = results[dataset_one_value] * self.gauss(data_value[i], mean, standart_dev)
        return results
        """
        calc_prob = {}
        for (classValue, classModels) in train_value.items():
            calc_prob[classValue] = 1
            for i in range(len(classModels)):
                (mean_numbers, standart_deviation, _) = classModels[i]
                result = data_value[i]
                calc_prob[classValue] *= self.gauss(result, mean_numbers, standart_deviation)
        return calc_prob

    def gauss(self, train_data, mean_value, standart_derivation_value):
        """
        Gauss olasılık dağılım hesaplaması yapılır. Bu hesaplama train data için yapılmaktadır. Test
        data için bu işlem yapılmaz.
        :param x:
        :param mean:
        :param stdev:
        :return:
        """
        if standart_derivation_value == 0.0:
            if train_data == mean_value:
                return 1.0
            else:
                return 0.0
        first_column = exp(-((train_data - mean_value) ** 2 / (2 * standart_derivation_value ** 2)))
        return  first_column * (1 / (sqrt(2 * pi) * standart_derivation_value))

    def naive_bayes(self, dataset, n_folds):
        """
        Bayes ana algoritmasıdır. İlk önce n_folds değerine göre veri kümesi parçalara bölünür.
        Her bir parçada gezinerek train ve test veri kümeleri bulunur. Yani veri kümesi k_cross değerlerine göre
        train ve test olarak iki parçaya ayrılır.
        :param dataset:
        :param n_folds:
        :return:
        """
        folds = self.k_cross(dataset, n_folds)
        f1_values = []
        predicted = []
        for fold in folds:
            train = list(folds)
            # Train için gerekli olan listedir. n_fold parametresine göre
            # ayrılan küçük parça kadar olmalıdır.
            train.remove(fold)
            train = sum(train, [])
            # Test değerleri ilk başta boş olarak oluşturulur.
            test = []
            # n_fold değerine göre ayrılan veri kümesindeki küçük parçalarda teker teker gezilmesi.
            for x in fold:
                test_x = list(x)
                #  Küçük parçalarda gezinerek test değerlerine eklenir, sonra o değerin eklenildiğinin
                # tahmin yapılırken anlaşılması için o değere None atanır.
                test.append(test_x)
                if self.n >= 3:
                    test_x[-1] = None
            # Değerlerin tahmin edilmesi için bayes metodunun çağrılması
            if self.n == 1:
                # PCA TOOL'UNUN KULLANILMASI
                pca = PCA(n_components=n_folds)
                test_pca = pca.fit_transform(test)
                train_pca = pca.fit_transform(train)
                test_pca, train_pca = self.controlWrongValues(len(test_pca), len(train_pca), test_pca, train_pca, test, train, 4)
                predicted = self.apply_bayes(train_pca, test_pca)
            elif self.n == 2:
                # LDA TOOL'UNUN KULLANILMASI
                lda = LDA(n_components=1)
                test_lda = lda.fit_transform(self.X_test, self.Y_test)
                train_lda = lda.fit_transform(self.X_train, self.Y_train)
                test_lda, train_lda = self.controlWrongValues(len(test), len(test), test_lda, train_lda, test, train, 0)
                predicted = self.apply_bayes(train_lda, test_lda)
            else:
                predicted = self.apply_bayes(train, test)
            # Belirtilen satır için veri kümesinde bulunan gerçek değerlerin alınması
            actual = [row[-1] for row in fold]
            # F1 Skorunun alınması
            f1_value_of_each_rows = helper.accuracy_metric(actual, predicted)
            f1_values.append(f1_value_of_each_rows)
        return f1_values

    def k_cross(self, data, n_folds):
        """
        Veri kümesi n_folds parametresi kadar bölünerek parçalara ayrılır.
        Ve liste içinde listeler şeklinde geri dönüş yapılır.
        :param data:
        :param n_folds:
        :return:
        """
        new_data = []
        temp_data = list(data)
        fold_size = int(len(data) / n_folds)
        for x in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(temp_data))
                fold.append(temp_data.pop(index))
            new_data.append(fold)
        return new_data


    def controlWrongValues(self, n1, n2, list1, list2, appendList1, appendList2, k):
        """
        Listedeki değerlerin float olup olmadığı kontrol eder.
        :param n1:
        :param n2:
        :param list1:
        :param list2:
        :param appendList1:
        :param appendList2:
        :param k:
        :return:
        """
        for i in range(n1):
            list1[i][k] = appendList1[i][4]
        for i in range(n2):
            list2[i][k] = appendList2[i][4]
        return list1, list2
