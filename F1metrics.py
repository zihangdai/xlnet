import csv

NO_TYPE_PENALTY_TYPES_1 = ["SPE1", "SPE2", "REL", "SIMI"]
NO_TYPE_PENALTY_TYPES_2 = ["SPE1", "SPE2", "SIMI"]
NO_TYPE_PENALTY_EQUI_2 = ["EQUI"]


class F1Metrics:

    def __init__(self, predicted_types, predicted_scores, target_types, target_scores, num_tokens):
        self._predicted_types = predicted_types
        self._predicted_scores = predicted_scores
        self._target_types = target_types
        self._target_scores = target_scores
        self._num_tokens = num_tokens
        self._total_num_tokens = sum(num_tokens)

    def print_statistics(self):
        print("[F1 Type], where alignment types need to match, but scores are ignored: {}".format(self.f1_type_match()))
        print("[F1 Score], where alignment type is ignored, but each alignment is penalized when scores do not match: {}".format(self.f1_score_match()))
        print("[F1 T+S],  where alignment types need to match and each alignment is penalized when scores do not match: {}".format(self.f1_all_match()))

    def f1_type_match(self):
        overlap = 0
        for i in range(len(self._predicted_types) - 1):
            if self._predicted_types[i] == self._target_types[i]:
                overlap += self._num_tokens[i]

        return self.count_f1(overlap, overlap)

    def f1_score_match(self):
        overlap = 0
        for i in range(len(self._predicted_scores) - 1):
            overlap += self._num_tokens[i] * (1 - abs(self._predicted_scores[i] - self._target_scores[i]) / 5)

        return self.count_f1(overlap, overlap)

    def f1_all_match(self):
        overlap = 0
        for pt, ps, tt, ts, tokens in zip(self._predicted_types, self._predicted_scores, self._target_types, self._target_scores, self._num_tokens):
            if pt == tt or self.is_special_case1(pt, ps, tt, ts) or self.is_special_case2(pt, ps, tt, ts):
                overlap += tokens * (1 - abs(ps - ts) / 5)

        return self.count_f1(overlap, overlap)

    def is_special_case1(self, predicted_type, predicted_score, target_type, target_score):
        if target_score <= 2 and predicted_score <= 2:
            if predicted_type in NO_TYPE_PENALTY_TYPES_1 and target_type in NO_TYPE_PENALTY_TYPES_1 and predicted_type != target_type:
                return True

        return False

    def is_special_case2(self, predicted_type, predicted_score, target_type, target_score):
        if predicted_score == 4 or target_score == 4:
            if predicted_type in NO_TYPE_PENALTY_EQUI_2 and target_score >= 4 and target_type in NO_TYPE_PENALTY_TYPES_2 \
            or target_type in NO_TYPE_PENALTY_EQUI_2 and predicted_score >= 4 and predicted_type in NO_TYPE_PENALTY_TYPES_2:
                return True
        return False

    def count_f1(self, overlap_gs, overlap_sys):
        precision = overlap_gs / self._total_num_tokens
        recall = overlap_sys / self._total_num_tokens

        return 2 * precision * recall/ (precision + recall)


def load_files_to_metrics(predicted_path, target_path):
    predicted_file = open(predicted_path)
    target_file = open(target_path)

    predicted_types, predicted_scores = read_predicted(predicted_file)

    target_types, target_scores, tokens = read_target(target_file)

    return F1Metrics(predicted_types, predicted_scores, target_types, target_scores, tokens)


def read_predicted(predicted_file):
    reader = csv.reader(predicted_file, delimiter='\t')
    next(reader)
    predicted_types = []
    predicted_scores = []

    for row in reader:
        print(row)
        type, score = row[1].split("-")
        predicted_types.append(type)
        predicted_scores.append(int(score))

    return predicted_types, predicted_scores

def read_target(target_file):
    reader = csv.reader(target_file, delimiter='\t')
    target_types = []
    target_scores = []
    tokens = []

    for row in reader:
        type, score = row[0].split("-")
        target_types.append(type)
        target_scores.append(int(score))
        tokens.append(int(row[3]))

    return target_types, target_scores, tokens

metrics = load_files_to_metrics("pred/ists/images-8000/ists.tsv", "ists/images/test.tsv")
metrics.print_statistics()
