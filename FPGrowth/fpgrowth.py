# Gökhan Has - 161044067
# CSE 454 - DATA MINING
# ASSIGNMENT 03
# fpgrowth.py

from TreeNode import TreeNode
import itertools
from collections import Counter
from itertools import chain

class FPGrowth(object):
    def __init__(self, transactions, support_threshold, verbose=True):
        # Parametreleri kaydet.
        self.verbose = verbose
        self.transactions = transactions
        self.threshold = support_threshold

        if self.verbose:
            print("Transactions", self.transactions)

    def build_fptree(self, data=None):
        """
        Bu methodda fpgrowth ağacı oluşturulur.
        :param data: Eğer data parametresi verilirse transaction olarak data parametresi kullanılır.
        :return: oluşturulan ağaç geri dönüş değeri olur.
        """
        if data is not None:
            self.transactions = data

        # Her bir itemin tüm transactionlarda ki sayısını bulalım.
        item_frequencies = Counter(list(chain(*self.transactions)))
        if self.verbose:
            print("Item frequencies", dict(item_frequencies))

        # Bulunan item sayılarının threshold altında kalanlarını çıkaralım
        frequent_items = {x: item_frequencies[x] for x in item_frequencies if
                          item_frequencies[x] >= self.threshold}
        if self.verbose:
            print("Threshold : {} Filtered items : {} ".format(self.threshold, frequent_items))

        # Eğer threshold sonrası bulunan itemlerın sayısı 0 ise geri döner
        if len(frequent_items) == 0:
            print("Item frequency degeri verilen threshold değerine({}) eşit veya yüksek item bulunamadı.".format(
                self.threshold))
            return

        # Her bir itemin frequency değerine göre sıralı bir listede elimizde bulunması gerekmektedir.
        frequent_sorted_items = sorted(frequent_items, key=lambda k: frequent_items[k], reverse=True)
        if self.verbose:
            print("Frequent Sorted Items: ", frequent_sorted_items)

        # her bir itemin parentını tutabilmek için none şeklinde bir parametre daha yanlarına ek
        header = {k: [v, None] for k, v in frequent_items.items()}

        # Root noktaımızı oluşturalım.
        root_node = TreeNode(None, None, None)

        # Verideki her bir transaction için aşağıdaki adımlar uygulanır.
        for transaction in self.transactions:
            # Transaction içindeki frequent_sorted_items içerisinde bulunan sıra ile aranır.
            frequent_sorted_transaction_items = [item for item in frequent_sorted_items if item in transaction]
            if len(frequent_sorted_transaction_items) > 0:
                # Ağacımızı update edelim.
                update_tree(frequent_sorted_transaction_items, header, root_node)
        return header

    def find_frequent_patterns(self, tree, support_data, f):
        # İlk önce oluşturduğumuz treenin ilk nodelarını alıyoruz.
        x = [r[0] for r in sorted(tree.items(), key=lambda r: r[1][0])]
        # Her bir node için  o node ile ilgili olan diğer nodelar bulunur ve bu nodeun sayısı eklenerek path elde edilir
        # Daha sonra sadece bulunan o path için bir ağaç oluşturulup o ağaç üzerinde ilerlenerek birbiri ile ilişki olan
        # itemlar support_data parametresi her fonksyina gönderildiği için bu parametre içerisinde saklanır.
        for i in x:
            ss = support_data.copy()
            ss.add(i)
            f[tuple(ss)] = tree[i][0]
            data = self.find_prefix_path(tree[i][1])
            if len(data) > 0:
                new_header = self.build_fptree(data)
                if new_header is not None:
                    self.find_frequent_patterns(new_header, ss, f)

    def find_prefix_path(self, node):
        data = {}
        while node is not None:
            path = self.ascend_path(node)
            if len(path) > 0:
                data[frozenset(path)] = node.count
            node = node.nodelink
        return data

    def ascend_path(self, node):
        path = []
        if node.parent is not None:
            while node.parent.value is not None:
                path.append(node.parent.value)
                node = node.parent
        return path

    def generate_association_rules(self, support_data, confidence):
        """
        Oluşturulan support data objesindeki her bir eleman için
        :param support_data:
        :param confidence:
        :return:
        """
        confidence_data = {}
        for data in support_data:
            if len(data) > 1:
                list_data = list(data)
                subset = itertools.chain(*[itertools.combinations(list_data, i + 1) for i, a in enumerate(list_data)])
                list_data = set(list_data)
                for set_data in subset:
                    set_data = set(set_data)
                    sd = tuple(list_data.difference(set_data))
                    if len(sd) > 0:
                        set_data = tuple(set_data)
                        if set_data in support_data:
                            cd = support_data[data]
                            if cd >= confidence:
                                confidence_data[tuple([set_data, sd])] = cd
        return confidence_data


def update_tree(items, header, node, count=1):
    """
    Ağacı recursive olarak update eder.
    :param items:
    :param header:
    :param node:
    :param count:
    :return:
    """
    # Gönderilen itemlardan ilki gönderilen node içerisinde bulunuyor mu ?
    if items[0] in node.children:
        # Bulunuyorsa count kadar eklenir.
        node.children[items[0]].count += count
    else:
        # Bulunmuyorsa nodeun childına yeni node eklenir.
        node.children[items[0]] = TreeNode(items[0], count, node)
        # Eklenen nodun parentı yoksa parentına ekle
        if header[items[0]][1] is None:
            header[items[0]][1] = node.children[items[0]]
        else:
            # Parentı varsa parentın childı olarak ekle.
            _node = header[items[0]][1]
            while _node.nodelink is not None:
                _node = _node.nodelink
            _node.nodelink = node.children[items[0]]
    if len(items) > 1:
        # Update recursively
        update_tree(items[1:], header, node.children[items[0]], count)
