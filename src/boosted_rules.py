import numpy as np


class BoostedRuleLearner:
    def __init__(self):
        self.rules = []
        self.D = None

    def predict(self, X):
        """
        DNF prediciton
        """

        preds = np.zeros(X.shape[0],)
        for rule in self.rules:
            preds += rule.predict(X) * rule.C_R

        return preds > 0


class Rule:
    """
    A rule calssifies an example based on conditions
    """
    def __init__(self):
        self.conditions = []
        self.C_R = 0

    def __str__(self):
        output = ''
        for condition in self.conditions:
            output += str(condition) + '\n'

        return output

    def add_condition(self, feature, operation, value, feature_map):
        self.conditions.append(
            Condition(feature, operation, value, feature_map)
        )

        # reset Z_tilda when adding new condition
        self.Z_tilda = None

    def predict(self, X, return_idx=False):
        """
        Take conjunction of conditions and make
        prediction for rule
        """
        if len(self.conditions) < 1:
            raise Exception("No conditions for rule, add conditions!")

        # sieve approach to gradually remove indices
        positive_cases = set(range(X.shape[0]))
        for condition in self.conditions:
            outputs = set(list(condition.classify(X)))
            positive_cases = positive_cases.intersection(outputs)

        if return_idx:
            return list(positive_cases)

        output = np.zeros(X.shape[0])
        output[list(positive_cases)] = 1

        return output

    def _get_design_matrices(self, X, y, D):
        """
        Helper funtion to calculate Ws for grow rule
        and Vs for prune rule
        """
        preds = self.predict(X)

        W_plus_idx = np.where((preds == 1) & (y == 1))
        W_minus_idx = np.where((preds == 1) & (y == 0))

        return np.sum(D[W_plus_idx]), np.sum(D[W_minus_idx])

    def calc_C_R(self, plus, minus, D):
        self.C_R = 0.5 * np.log((plus + (1 / (2 * len(D)))) /
                                (minus + 1 / (2 * len(D))))


class Condition:
    """
    Models conditions for a feature that make up a rule
    """
    def __init__(self, feature, operation, value, feature_map=None):
        self.feature = feature
        self.operation = operation
        self.value = value
        self.feature_map = feature_map

    def __str__(self):
        out = ''
        if self.feature_map:
            out += self.feature_map[self.feature]
        else:
            out += str(self.feature)

        return out + ' ' + self.operation + ' ' + str(self.value)

    def __eq__(self, other):
        return self.feature == other.feature and \
            self.operation == other.operation and \
            self.value == other.value

    def classify(self, X, return_idx=True):
        """
        Apply condition and return indices where condition
        is satisifed
        """
        logic = 'X[:, self.feature] {} self.value'.format(self.operation)
        output = np.where(eval(logic))

        if not return_idx:
            return eval(logic)

        return output[0]
