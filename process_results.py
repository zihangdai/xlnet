import csv
import os

def process_results(predictions, real, dataset_name):

    assert os.path.exists(predictions), "Prediction file {} does not exists".format(predictions)
    assert os.path.exists(real), "Real file {} does not exists".format(real)

    predictions_file = open(predictions)
    real_file = open(real)

    real_classes = get_classes_from_real(real_file)
    predicted_classes = get_classes_from_predicted(predictions_file)

    assert len(predicted_classes) == len(real_classes) , "Prediction file size {} is not equal real file size {}"\
       .format(len(predicted_classes), len(real_classes))

    true_positive, predicted, real = get_true_predicted_real(real_classes, predicted_classes)
    metrics = calculate_metrics(true_positive, predicted, real)
    print(metrics)
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

def get_true_predicted_real(real_classes, predicted_classes):
    true_positive = [0] * len(set(real_classes))
    predicted = [0] * len(set(real_classes))
    real = [0] * len(set(real_classes))

    idx = 0

    for i in set(real_classes):
        for j in range(len(predicted_classes) - 1):
            if real_classes[j] == i and predicted_classes[j] == i:
                true_positive[idx] = true_positive[idx] + 1
            if real_classes[j] == i:
                real[idx] = real[idx] + 1
            if predicted_classes[j] == i:
                predicted[idx] = predicted[idx] + 1

        idx += 1

    return true_positive, predicted, real

def calculate_metrics(true_positive, predicted, real):
    precision = [0] * len(real)
    recall = [0] * len(real)
    f1_score = [0] * len(real)

    for i in range(len(real) - 1):
        precision[i] = 0 if predicted[i] == 0 else true_positive[i] / predicted[i]
        recall[i] = 0 if real[i] == 0 else true_positive[i] / real[i]
        f1_score[i] = 0 if precision[i] == 0 and recall[i] == 0 else 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    precision_avg = 0
    recall_avg = 0
    f1_score_avg = 0

    for i in range(len(real) - 1):
        precision_avg += precision[i] / len(real)
        recall_avg += recall[i] / len(real)
        f1_score_avg = f1_score_avg + f1_score[i] / len(real)

    return {"precision": precision_avg,
            "recall": recall_avg,
            "f1_score": f1_score_avg}

def save_metrics(metrics, dataset_name):
    with open(dataset_name, 'w', newline='') as myfile:
        wr = csv.DictWriter(myfile, delimiter='\t', lineterminator='\n', fieldnames=metrics.keys())
        wr.writeheader()
        wr.writerow(metrics)

process_results("ists/headlines/test.tsv", "pred/ists/headlines/ists.tsv", "headlines")