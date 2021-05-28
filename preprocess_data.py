import re
import csv
import os

RESULTS = ["EQUI-5", "OPPO-1", "OPPO-2", "OPPO-3", "OPPO-4",
                "SPE1-1", "SPE1-2", "SPE1-3", "SPE1-4",
                "SPE2-1", "SPE2-2", "SPE2-3", "SPE2-4",
                "SIMI-1", "SIMI-2", "SIMI-3", "SIMI-4",
                "REL-1", "REL-2", "REL-3", "REL-4"]
def preprocess_data(input, output, test_data = False):

    assert os.path.exists(input), "Input file: {} does not exists".format(input)

    file = open(input, 'r+')
    data = file.read()
    results = []
    for alignments in re.finditer("(<alignment>(.|\\n|\\r\\n)*?<\\/alignment>)", data):
        alignments = re.split('\\n', alignments.group(0))
        for alignment in alignments[1:-1]:
            elements = alignment.split("//")
            if '-not aligned-' in [x.strip() for x in elements[3].split("<==>")]:
                continue
            type = elements[1].replace(" ", "").split("_")[0]
            score = elements[2].replace(" ", "")
            score = '0' if elements[2].replace(" ", "") == 'NIL' else score
            texts = elements[3].split("<==>")
            textA = texts[0].strip()
            textB = texts[1].strip()
            tokens = elements[0].split("<==>")
            num_tokens = min(len(list(map(int, tokens[0].split()))), len(list(map(int, tokens[1].split()))))
            res = [type+"-"+score, textA, textB, num_tokens]
            if res[0] not in RESULTS:
                print("WARRNING: Wrong input data: {}".format(res[0]))
                continue
            results.append(res)
    with open(output, 'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter='\t', lineterminator='\n')
        wr.writerows(results)


preprocess_data("data/task2/semeval/answers-students/answers_students_test_gs.wa", "ists/answers-students/test.tsv")
preprocess_data("data/task2/semeval/answers-students/answers_students_train_gs.wa", "ists/answers-students/train.tsv")
preprocess_data("data/task2/semeval/headlines/headlines_test_gs.wa", "ists/headlines/test.tsv")
preprocess_data("data/task2/semeval/headlines/headlines_train_gs.wa", "ists/headlines/train.tsv")
preprocess_data("data/task2/semeval/images/images_phrases_test_gs.wa", "ists/images/test.tsv")
preprocess_data("data/task2/semeval/images/images_phrases_train_gs.wa", "ists/images/train.tsv")
