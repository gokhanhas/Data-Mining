# Gökhan Has - 161044067
# CSE 454 - DATA MINING
# ASSIGNMENT 03
# main.py

from fpgrowth import FPGrowth
import numpy as np
import pandas as pd

def main(data_path, support_threshold, confidence, verbose):
    """
    Eğer verilen data_path parametresinde bir dosya bulunuyorsa o dosyayı yükler. Eger bulunmuyorsa
    number_of_transaction ve number_of_item parametreleri ile bir veri seti oluşturulur.
    number_of_transaction: toplam işlem(satır) sayısını göstermektedir.
    number_of_item: birbirinden farklı toplam ürün sayısını göstermektedir.
    Veri kümesi oluşturulurken her bir transaction(işlem) eşit sayıda ürün içermeyecek şekilde oluşturulmuştur.
    :param confidence:
    :param data_path: veri kümesinin bulunduğu dizin
    :param support_threshold: FpGrowth algoritması için Destek aralığı
    :param verbose: Debug mesajlarının ekrana bastırılması ile ilgili bool değer
    :return:
    """

    df = pd.read_csv(data_path, index_col=0)
    transactions = [d[~np.isnan(d)] for d in df.values] if True else df.values
    print(transactions)
    # Fpgrowth
    fpgrowth = FPGrowth(transactions=transactions, support_threshold=support_threshold, verbose=verbose)
    tree = fpgrowth.build_fptree()
    support_data = {}
    fpgrowth.find_frequent_patterns(tree, set([]), support_data)
    confidence_data = fpgrowth.generate_association_rules(support_data, confidence)
    print("support data", support_data)
    #print("confidence data", confidence_data)


if __name__ == '__main__':
    path = "data_v2.csv"
    main(path, .2, 3, True)