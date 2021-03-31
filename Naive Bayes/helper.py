# GÖKHAN HAS - 161044067
# CSE 454 - DATA MINING
# ASSIGNMENT 04
# helper.py


def accuracy_metric(actual, predicted):
    tp, fp, fn, tn, trues = 0, 0, 0, 0, 0
    for i in range(len(actual)):
        if actual[i] == 1 and predicted[i] == 1:
            tp += 1
            trues += 1
        elif actual[i] == 0 and predicted[i] == 1:
            fp += 1
        elif actual[i] == 1 and predicted[i] == 0:
            fn += 1
        else:
            tn += 1
            trues += 1
    if tp + fp == 0 or tp + fn == 0:
        return accuracy_metric_2(actual, predicted)
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

def accuracy_metric_2(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual))
