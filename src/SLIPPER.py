import copy
import random 
from enum import Enum

import numpy as np
from sklearn.model_selection import train_test_split


class SLIPPER:
    def __init__(self):
        self.rules = []
        self.D = None
        self.Z = None
        self.grow_idx = None
        self.prune_idx = None

    def __make_candidate(self, X, y, curr_rule, feat, A_c):
        """
        Make candidate rule based off new condition and
        existing rule
        """
        # TODO: smarter with list of conditions
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

            if curr_rule.Z_tilda >= candidate_rule.Z_tilda or len(negative_coverage) == 0:
                stop_condition = True
            else:
                curr_rule = copy.deepcopy(candidate_rule)

        return curr_rule

    def __prune_rule(self, X, y, rule):
        """
        Removes conditions from initial rule built on growth
        set minimizing objective formula
        """

        stop_condition = False
        curr_rule = copy.deepcopy(rule)
        curr_rule.prune_rule(X, y, self.D)

        while not stop_condition:
            candidate_rules = []
            for condition in curr_rule.conditions:
                R_prime = copy.deepcopy(curr_rule)
                R_prime.prune_rule(X, y, self.D, condition)
                candidate_rules.append(R_prime)

            prune_objs = [x.pobj for x in candidate_rules]
            best_prune = candidate_rules[
                prune_objs.index(min(prune_objs))
            ]

            if curr_rule.pobj > best_prune.pobj:
                curr_rule = copy.deepcopy(best_prune)
            else:
                stop_condition = True

        return curr_rule

    def make_default_rule(self, X, y):
        """
        Function to make default rule
        """
        default_rule = Rule()
        features = random.choices(
            list(range(X.shape[1])),
            k=random.randint(2, X.shape[1])
        )

        for i, x in enumerate(features):
            if i % 2:
                default_rule.add_condition(
                    x, "<=", max(X[:, x])
                )
            else:
                default_rule.add_condition(
                    x, ">=", min(X[:, x])
                )
        
        return default_rule

    def add_rule_or_default(self, X, y, learned_rule):
        """
        add rule or default
        """
        rules = [self.make_default_rule(X, y), learned_rule]
        scores = [x.calc_eq_5(X, y, self.D) for x in rules]
        self.rules.append(rules[scores.index(min(scores))])

    def fit(self, X, y, T=10):
        """
        Main loop for training
        """
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

            self.add_rule_or_default(X, y, rule_t)

    def predict(self, X):
        """
        Find conjunction of all rules
        """

        preds = np.zeros(X.shape[0],)
        for rule in self.rules:
            preds += rule.predict(X)

        return preds


class Rule:
    """
    A rule calssifies an example based on conditions
    """
    def __init__(self):
        self.conditions = []
        self.Z_tilda = 0
        self.pobj = 0  # prune rule objective value
        self.C_R = 0
        self._pruned = False  # TODO: Replace with enum (grow, prune, default)

    def __str__(self):
        output = ''
        for condition in self.conditions:
            output += str(condition) + '\n'

        return output

    def add_condition(self, feature, operation, value):
        self.conditions.append(
            self.Condition(feature, operation, value)
        )
        # reset Z_tilda when adding new condition
        self.Z_tilda = None

    def prune_rule(self, X, y, D, condition=None):
        """
        Convert rule to a pruned rule
        """
        if condition:
            self.conditions.remove(condition)

        self._pruned = True

        self.prune_rule_obj(X, y, D)

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

        output = np.ones(X.shape[0]) * -1
        output[list(positive_cases)] = 1

        return output

    def __get_design_matrices(self, X, y, D):
        """
        Helper funtion to calculate Ws for grow rule
        and Vs for prune rule
        """
        preds = self.predict(X)

        W_plus_idx = np.where((preds == 1) & (y == 1))
        W_minus_idx = np.where((preds == 1) & (y == 0))

        return np.sum(D[W_plus_idx]), np.sum(D[W_minus_idx])

    def calc_C_R(self, plus, minus, D):
        self.C_R = 0.5 * np.log((plus + (1 / (2 * len(D)))) \
            / (minus + 1 / (2 * len(D))))

    def grow_rule_obj(self, X, y, D):
        """
        Score equation (6) and get confidence
        from equation (4)
        """
        W_plus, W_minus = self.__get_design_matrices(X, y, D)

        self.calc_C_R(W_plus, W_minus, D)

        self.Z_tilda = np.sqrt(W_plus) - np.sqrt(W_minus)

    def prune_rule_obj(self, X, y, D):
        """
        Objective function for prune rule routine
        """
        V_plus, V_minus = self.__get_design_matrices(X, y, D)

        # TODO: This is really not safe to update C_R like this 
        # between two rules
        self.pobj = (1 - V_plus - V_minus) + V_plus * np.exp(-self.C_R) + V_minus * np.exp(self.C_R)

    def calc_eq_5(self, X, y, D):
        W_plus, W_minus = self.__get_design_matrices(X, y, D)
        return 1 - np.square(np.sqrt(W_plus) - np.sqrt(W_minus))

    class State(Enum):
        GROW = 1
        PRUNED = 2
        DEFAULT = 3

    class Condition:
        """
        Models conditions for a feature that make up a rule
        """
        def __init__(self, feature, operation, value):
            self.feature = feature
            self.operation = operation
            self.value = value

        def __str__(self):
            return str(self.feature) + ' ' + self.operation + \
                ' ' + str(self.value)

        def __eq__(self, other):
            return self.feature == other.feature and \
                self.operation == other.operation and \
                    self.value == other.value

        def classify(self, X):
            """
            Apply condition and return indices where condition
            is satisifed
            """
            logic = 'X[:, self.feature] {} self.value'.format(self.operation)
            output = np.where(eval(logic))

            return output[0]
