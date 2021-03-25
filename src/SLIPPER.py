import numpy as np
from sklearn.model_selection import train_test_split


class SLIPPER:
    def __init__(self):
        self.rules = {}
        self.D = None
        self.Z = None
        self.grow_idx = None
        self.prune_idx = None

    def __process_rule(self, X, rule):
        """
        Takes rule in form
        {feature: (["==", ">=", "<="], value), ... }
        and returns predictions
        """

        for 
        return

    def C_R(self, temp_rule_dict, n):
        """
        Function to score confidence of rule
        Equation (4)
        """
        return 0.5 * np.log((temp_rule_dict['W_plus'] + (1 / (2 * n))) \
            / (temp_rule_dict['W_minus'] + 1 / (2 * n)))

    def __grow_obj_calc(self, y_grow, output_idx):
        """
        Compute objective funciton for grow rule routine and
        store W_plus and W_minus

        grow_output:
            {
                'W_plus': float,
                'W_minus': float,
                'Z_tilda': float
            }
        """
        grow_output = {}
        W_plus_idx = np.where(y_grow[output_idx] == 1)
        W_minus_idx = np.where(y_grow[output_idx] == 0)

        grow_output['W_plus'] = np.sum(self.D[W_plus_idx])
        grow_output['W_minus'] = np.sum(self.D[W_minus_idx])

        grow_output['Z_tilda'] = np.sqrt(grow_output['W_plus']) \
            - np.sqrt(grow_output['W_minus'])

        return grow_output

    def __make_candidate(self, X, y, feat, A_c):
        """
        Helper function to make a candidate rule based off
        a value

        optimal_candidate:
            {
                'W_plus': float,
                'W_minus': float,
                'Z_tilda': float,
                'operation': string
            }
        """
        # # Get indices to build W_plus and W_minus
        # output_gte_idx = np.where(X[:, feat] >= A_c)[0]
        # output_lte_idx = np.where(X[:, feat] <= A_c)[0]
        output_gte_idx = self.__process_rule(X, rule)
        output_lte_idx = self.__process_rule(X, rule)

        output_lte = self.__grow_obj_calc(y, output_lte_idx)
        output_gte = self.__grow_obj_calc(y, output_gte_idx)

        optimal_candidate = output_lte \
            if output_lte['Z_tilda'] > output_gte['Z_tilda'] \
            else output_gte

        optimal_candidate['operation'] = '>=' \
            if optimal_candidate == output_gte['Z_tilda'] else '<='

        return optimal_candidate

    def __update_candidate(self, curr_best, candidates, feat, m):
        """
        Helper function to update and build candidate rule
        if it improves Z_tilda
        """

        # find best performing rule from batch
        Zs = [x['Z_tilda'] for x in candidates]
        candidate_dict = candidates[candidates.index(max(Zs))]

        # check if there is no current best
        if not curr_best:
            candidate_dict['feat'] = feat
            return candidate_dict

        if candidate_dict['Z_tilda'] > curr_best['Z_tilda']:
            candidate_dict['C_R'] = self.C_R(candidate_dict, m)

            self.Z = candidate_dict['Z_tilda']
        else:
            candidate_dict = None

        return candidate_dict

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

        prev_rule, candidate_rule = None, None
        features = [i for i in range(X.shape[1])]
        rule = []

        while prev_rule != candidate_rule:
            for feat in features:
                pivots = np.percentile(X[:, feat], range(0, 100, 10),
                                       interpolation='midpoint')
                feat_candidates = [
                    self.__make_candidate(X, y, feat, A_c)
                    for A_c in pivots
                ]

                candidate_rule = self.__update_candidate(
                    candidate_rule,
                    feat_candidates,
                    feat,
                    X.shape[0]
                )

            if candidate_rule:
                rule.append(candidate_rule)

        return {}

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
        idx = np.array([i for i in range(m)])

        for _ in range(T):
            X_grow, X_prune, y_grow, y_prune, grow_idx, prune_idx = \
                train_test_split(X, y, idx, test_size=0.33)

            # save actual idx of entry to update distributions
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
