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
        {feature: (["==", ">=", "<="], value, confidence)}
        and returns predictions
        """
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
        """
        grow_output = {}
        W_plus_idx = np.where(y_grow[output_idx] == 1)
        W_minus_idx = np.where(y_grow[output_idx] == 0)

        grow_output['W_plus'] = np.sum(self.D[W_plus_idx])
        grow_output['W_minus'] = np.sum(self.D[W_minus_idx])

        grow_output['Z_tilda'] = np.sqrt(grow_output['W_plus']) \
            - np.sqrt(grow_output['W_minus'])

        return grow_output

    def __make_candidate(self, X, y, A_c):
        """
        Helper function to make a candidate rule based off 
        a value
        """
        # Get indices to build W_plus and W_minus
        output_gte_idx = np.where(X[:, feat] >= A_c)[0]
        output_lte_idx = np.where(X[:, feat] <= A_c)[0]

        output_lte = self.__grow_obj_calc(y, output_lte_idx)
        output_gte = self.__grow_obj_calc(y, output_gte_idx)

        optimal_candidate = max(output_lte['Z_tilda'], 
                                output_gte['Z_tilda'])

        optimal_candidate = output_lte \
            if output_lte['Z_tilda'] > output_gte['Z_tilda'] \
            else output_gte

        optimal_candidate['operation'] = '>=' \
            if optimal_candidate == output_gte['Z_tilda'] else '<='

        candidate_rule =

        return optimal_candidate, 

    def __update_candidate(self, candidate_rule, optimal_candidate, m):
        """
        Helper function to update and build candidate rule 
        if it improves Z_tilda
        """
        new_candidate_rule = candidate_rule

        if new_candidate_rule and optimal_candidate > best:
            C_R = self.C_R(optimal_candidate, m)

            new_candidate_rule = (feat, operation, A_c, C_R)

            z_delta = (self.Z - optimal_candidate) / self.Z \
                if self.Z else z_delta

            self.Z = best

        return new_candidate_rule

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

        features = [i for i in range(X.shape[1])]
        candidate_rule = None
        z_delta = 100
        rule = []

        # iterate while improvement is less than tolerance
        while z_delta >= tol or not candidate_rule:
            for feat in features:
                for _ in range(3):  # TODO: loop based on con_tol
                    candidates = np.percentile(X[:, feat], range(0, 100, 25),
                                               interpolation='midpoint')
                    for A_c in candidates:
                        optimal_candidate = self.__make_candidate(X, y, A_c)

                        

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

        for t in range(T):
            X_grow, X_prune, y_grow, y_prune, grow_idx, prune_idx= \
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
