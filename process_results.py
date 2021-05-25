import csv
import os
from sklearn.metrics import f1_score, recall_score, precision_score

def process_results(predictions, real, dataset_name):

    assert os.path.exists(predictions), "Prediction file {} does not exists".format(predictions)
    assert os.path.exists(real), "Real file {} does not exists".format(real)

    predictions_file = open(predictions)
    real_file = open(real)

    real_classes = get_classes_from_real(real_file)
    predicted_classes = get_classes_from_predicted(predictions_file)


    assert len(predicted_classes) == len(real_classes) , "Prediction file size {} is not equal real file size {}"\
       .format(len(predicted_classes), len(real_classes))

    metrics = calculate_metrics(real_classes, predicted_classes)

    save_metrics(metrics, dataset_name)

def get_classes_from_real(file):
    real_classes = []
    reader = csv.reader(file, delimiter='\t')
    next(reader)
    for row in reader:
        real_classes.append(row[1])

    return real_classes

def get_classes_from_predicted(file):
    reader = csv.reader(file, delimiter='\t')
    predicted_classes = []

    for row in reader:
        predicted_classes.append(row[0])

    return predicted_classes

def calculate_metrics(real_classes, predicted_classes):
    precision_micro = precision_score(real_classes, predicted_classes, average='micro')
    precision_macro = precision_score(real_classes, predicted_classes, average='macro')
    precision_weighted = precision_score(real_classes, predicted_classes, average='weighted')

    recall_micro = recall_score(real_classes, predicted_classes, average='micro')
    recall_macro = recall_score(real_classes, predicted_classes, average='macro')
    recall_weighted = recall_score(real_classes, predicted_classes, average='weighted')

    f1_micro = f1_score(real_classes, predicted_classes, average='micro')
    f1_macro = f1_score(real_classes, predicted_classes, average='macro')
    f1_weighted = f1_score(real_classes, predicted_classes, average='weighted')

    metrics = {
        "precision_micro": round(precision_micro, 5),
        "precision_macro": round(precision_macro, 5),
        "precision_weighted": round(precision_weighted, 5),

        "recall_micro": round(recall_micro, 5),
        "recall_macro": round(recall_macro, 5),
        "recall_weighted": round(recall_weighted, 5),

        "f1_micro": round(f1_micro, 5),
        "f1_macro": round(f1_macro, 5),
        "f1_weighted": round(f1_weighted, 5)
    }

    return metrics

def save_metrics(metrics, dataset_name):
    with open(dataset_name + '.tsv', 'w', newline='') as myfile:
        wr = csv.DictWriter(myfile, delimiter='\t', lineterminator='\n', fieldnames=metrics.keys())
        wr.writeheader()
        wr.writerow(metrics)


process_results("ists/images/test.tsv", "pred/ists/images-8000/ists.tsv", "images")