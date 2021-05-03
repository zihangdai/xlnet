import re
import csv

def preprocess_data(input, output):
    file = open(input, 'r+')
    data = file.read()
    results = []
    for alignments in re.finditer("(<alignment>(.|\\n|\\r\\n)*?<\\/alignment>)", data):
        alignments = re.split('\\n', alignments.group(0))
        for alignment in alignments[1:-1] :
            elements = alignment.split("//")
            if '-not aligned-' in [x.strip() for x in elements[3].split("<==>")]:
                continue
            type = elements[1].replace(" ", "")
            score = elements[2].replace(" ", "")
            score = '0' if elements[2].replace(" ", "") == 'NIL' else score
            texts = elements[3].split("<==>")
            textA = texts[0].strip()
            textB = texts[1].strip()
            res = [type+"-"+score, textA, textB]
            results.append(res)
    with open(output, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(results)


preprocess_data("test.wa", "test.csv")
