import copy
import random
from enum import Enum

import numpy as np
from sklearn.model_selection import train_test_split

from src.boosted_rules import BoostedRuleLearner, Rule


class SLIPPER(BoostedRuleLearner):
    def __init__(self, features=None):
        super().__init__()
        self.Z = None
        if features is not None:
            self.feature_map = {i: feat for i, feat in enumerate(features)}
        else:
            self.feature_map = None

    def __make_candidate(self, X, y, curr_rule, feat, A_c):
        """
        Make candidate rule based off new condition and
        existing rule
        """

        # TODO: smarter with list of conditions
        # Get indices to build W_plus and W_minus
        gte_rule = copy.deepcopy(curr_rule)
        lte_rule = copy.deepcopy(curr_rule)
        eq_rule = copy.deepcopy(curr_rule)

        gte_rule.add_condition(feat, '>=', A_c, self.feature_map)
        lte_rule.add_condition(feat, '<=', A_c, self.feature_map)
        eq_rule.add_condition(feat, '==', A_c, self.feature_map)

        gte_rule.grow_rule_obj(X, y, self.D)
        lte_rule.grow_rule_obj(X, y, self.D)
        eq_rule.grow_rule_obj(X, y, self.D)

        optimal = max(eq_rule.Z_tilda, gte_rule.Z_tilda, lte_rule.Z_tilda)

        if optimal == gte_rule.Z_tilda:
            return gte_rule
        elif optimal == lte_rule.Z_tilda:
            return lte_rule
        else:
            return eq_rule

    def __grow_rule(self, X, y):
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
        curr_rule = SlipperRule()

        while not stop_condition:
            candidate_rule = copy.deepcopy(curr_rule)
            for feat in features:
                pivots = np.percentile(X[:, feat], range(0, 100, 10),
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

            if curr_rule.Z_tilda >= candidate_rule.Z_tilda or \
                    len(negative_coverage) == 0:
                stop_condition = True
            else:
                curr_rule = copy.deepcopy(candidate_rule)

        return curr_rule

    def __prune_rule(self, X, y, rule):
        """
        Removes conditions from initial rule built on growth
        set minimizing objective formula (eq 7)
        """

        stop_condition = False
        curr_rule = copy.deepcopy(rule)
        curr_rule.prune_rule(X, y, self.D)

        while not stop_condition:
            candidate_rules = []

            if len(curr_rule.conditions) == 1:
                return curr_rule

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
        default_rule = SlipperRule(rule_state=RuleState.DEFAULT)
        features = random.choices(
            list(range(X.shape[1])),
            k=random.randint(2, 8)  # arbitrary choice for max conditions
        )

        for i, x in enumerate(features):
            if i % 2:
                default_rule.add_condition(
                    x, "<=", max(X[:, x]), self.feature_map
                )
            else:
                default_rule.add_condition(
                    x, ">=", min(X[:, x]), self.feature_map
                )

        return default_rule

    def add_rule_or_default(self, X, y, learned_rule):
        """
        add rule or default
        """
        rules = [self.make_default_rule(X, y), learned_rule]
        scores = [x.calc_eq_5(X, y, self.D) for x in rules]
        self.rules.append(rules[scores.index(min(scores))])

    def update(self, X, y):
        """
        Function to update distributions
        """
        self.D /= np.exp(y * self.rules[-1].C_R)

        self.D /= np.sum(self.D)

    def fit(self, X, y, T=3):
        """
        Main loop for training
        """
        m = X.shape[0]
        self.D = np.array([1 / m for _ in range(m)])

        for _ in range(T):
            X_grow, X_prune, y_grow, y_prune = \
                train_test_split(X, y, test_size=0.33)

            rule_t = self.__grow_rule(X_grow, y_grow)
            rule_t = self.__prune_rule(X_prune, y_prune, rule_t)

            self.add_rule_or_default(X, y, rule_t)

            self.update(X, y)


class RuleState(Enum):
    GROW = 1
    PRUNE = 2
    DEFAULT = 3


class SlipperRule(Rule):
    def __init__(self, rule_state=RuleState.GROW):
        super().__init__()
        self.Z_tilda = -10000
        self.pobj = 0  # prune rule objective value
        self.state = rule_state

    def __str__(self):
        output = str(self.state) + '\n'
        for condition in self.conditions:
            output += str(condition) + '\n'

        return output

    def prune_rule(self, X, y, D, condition=None):
        """
        Convert rule to a pruned rule
        """
        if condition:
            self.conditions.remove(condition)

        self.state = RuleState.PRUNE

        self.prune_rule_obj(X, y, D)

    def grow_rule_obj(self, X, y, D):
        """
        Score equation (6) and get confidence
        from equation (4)
        """
        W_plus, W_minus = self._get_design_matrices(X, y, D)

        self.calc_C_R(W_plus, W_minus, D)

        self.Z_tilda = np.sqrt(W_plus) - np.sqrt(W_minus)

    def prune_rule_obj(self, X, y, D):
        """
        Objective function for prune rule routine
        """
        V_plus, V_minus = self._get_design_matrices(X, y, D)

        self.pobj = (1 - V_plus - V_minus) + V_plus * np.exp(-self.C_R) \
            + V_minus * np.exp(self.C_R)

    def calc_eq_5(self, X, y, D):
        W_plus, W_minus = self._get_design_matrices(X, y, D)
        return 1 - np.square(np.sqrt(W_plus) - np.sqrt(W_minus))
