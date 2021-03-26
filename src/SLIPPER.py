import copy

import numpy as np
from sklearn.model_selection import train_test_split


class Rule:
    """
    A rule calssifies an example based on conditions 
    """
    def __init__(self):
        self.conditions = []
        self.Z_tilda = 0
        self.C_R = 0

    def add_condition(self, feature, operation, value):
        self.conditions.append(
            self.Condition(feature, operation, value)
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

        output = np.zeros(X.shape[0])
        output[list(positive_cases)] = 1

        return output
    
    def grow_rule_obj(self, X, y, D):
        """
        Score equation (6) and get confidence 
        from equation (4)
        """
        preds = self.predict(X)

        W_plus_idx = np.where((preds == 1) & (y == 1))
        W_minus_idx = np.where((preds == 1) & (y == 0))

        W_plus = np.sum(D[W_plus_idx])
        W_minus = np.sum(D[W_minus_idx])

        self.C_R = 0.5 * np.log((W_plus + (1 / (2 * len(D)))) \
            / (W_minus + 1 / (2 * len(D))))

        self.Z_tilda = np.sqrt(W_plus) - np.sqrt(W_minus)

    class Condition:
        """
        Models conditions for a feature that make up a rule
        """
        def __init__(self, feature, operation, value):
            self.feature = feature
            self.operation = operation
            self.value = value

        def __str__(self):
            return str(self.feature) + ' ' + self.operation + ' ' + str(self.value)

        def classify(self, X):
            """
            Apply condition and return indices where condition
            is satisifed
            """
            logic = 'X[:, self.feature] {} self.value'.format(self.operation)
            output = np.where(eval(logic))

            return output[0]


class SLIPPER:
    def __init__(self):
        self.rules = {}
        self.D = None
        self.Z = None
        self.grow_idx = None
        self.prune_idx = None

    def __make_candidate(self, X, y, curr_rule, feat, A_c):
        """
        Make candidate rule based off new condition and 
        existing rule
        """
        # Get indices to build W_plus and W_minus
        gte_rule = copy.deepcopy(curr_rule)
        lte_rule = copy.deepcopy(curr_rule)

        gte_rule.add_condition(feat, '>=', A_c)
        lte_rule.add_condition(feat, '<=', A_c)

        gte_rule.grow_rule_obj(X, y, self.D)
        lte_rule.grow_rule_obj(X, y, self.D)

        optimal = max(gte_rule.Z_tilda, lte_rule.Z_tilda)

        if optimal == gte_rule.Z_tilda:
            return gte_rule
        else: 
            return lte_rule

    def __grow_rule(self, X, y, tol=0.01, con_tol=0.01):
        """
        Starts with empty conjunction of conditions and
        greddily adds rules to mazimize Z_tilda_t

        # Parameters
        #   X (np.array): Features for each observation
        #   y (np.array): Response feature
        #   tol: tolerance for when to end adding conditions to rule
        #   con_tol: condition tolerance for when to stop
        #            tuning condition for feature
        """

        stop_condition = False 
        features = list(range(X.shape[1]))
        curr_rule = Rule()

        while not stop_condition:
            candidate_rule = curr_rule
            for feat in features:
                # TODO: pivots should be actual search method
                pivots = np.percentile(X[:, feat], range(0, 100, 25),
                                       interpolation='midpoint')

                feature_candidates = [
                    self.__make_candidate(X, y, curr_rule, feat, A_c)
                    for A_c in pivots
                ]

                # get max Z_tilda and update candidate accordingly
                tildas = [x.Z_tilda for x in feature_candidates]
                if max(tildas) > candidate_rule.Z_tilda:
                    candidate_rule = feature_candidates[
                        tildas.index(max(tildas))
                    ]

            preds = candidate_rule.predict(X)
            negative_coverage = np.where((preds == y) & (y == 0))

            print(curr_rule.Z_tilda > candidate_rule.Z_tilda)
            if curr_rule.Z_tilda > candidate_rule.Z_tilda or \
                 len(negative_coverage) == 0:
                stop_condition = True
            else:
                curr_rule = copy.deepcopy(candidate_rule)

        return curr_rule

    def __prune_rule(self, X, y, rule):
        """
        Removes conditions from initial rule built on growth
        set minimizing objective formula
        """
        return {}

    def fit(self, X, y, T=1):
        """
        Main loop for training
        """

        rules = []
        m = X.shape[0]
        self.D = np.array([1 / m for _ in range(m)])
        idx = np.array(list(range(m)))

        for _ in range(T):
            X_grow, X_prune, y_grow, y_prune, grow_idx, prune_idx = \
                train_test_split(X, y, idx, test_size=0.33)

            # save actual index of entry to update distributions
            self.grow_idx = grow_idx
            self.prune_idx = prune_idx

            rule_t = self.__grow_rule(X_grow, y_grow)
            rule_t = self.__prune_rule(X_prune, y_prune, rule_t)

            rules.append(rule_t)

    def predict(self, X, y):
        """
        Function to return predictions given
        learned rules
        """
        return
