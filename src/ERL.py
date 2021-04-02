import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.boosted_rules import BoostedRuleLearner, Rule, Condition


class ERL(BoostedRuleLearner):
    def __init__(self, percentiles=10):
        super().__init__()
        self.LP = None
        self.conditions = []
        self.percentiles = 10

    def _make_measurement_matrix(self, X, y):
        """
        Make measurement matrix by adding conditions
        """
        preds = []
        for feat in range(X.shape[1]):
            pivots = np.percentile(X[:, feat], range(0, 100, self.percentiles),
                                   interpolation='midpoint')
            for pivot in pivots:
                for operation in ['>=', '<=', '==']:
                    condition = Condition(feat, operation, pivot)
                    preds.append(condition.classify(X, return_idx=False))
                    self.conditions.append(condition)

        measurement = np.vstack(preds)
        return measurement.T

    def _make_LP(self, measurement, y):
        """
        Run LP from ERL
        """
        # make measurement matrics
        A_p = measurement[np.where(y == 1)].astype(int)
        A_n = measurement[np.where(y != 1)].astype(int)

        # define model and other paramters
        m = gp.Model('rule-extraciton')
        w = m.addMVar(shape=measurement.shape[1], name="weights")
        psi_p = m.addMVar(shape=A_p.shape[0], name="psi_p")
        psi_n = m.addMVar(shape=A_n.shape[0], name="psi_n")

        # add model constraints
        m.addConstr(w <= 1.0)
        m.addConstr(w >= 0.0)
        m.addConstr(psi_p <= 1)
        m.addConstr(psi_p >= 0)
        m.addConstr(psi_n >= 0)
        m.addConstr(A_p @ w + psi_p >= 1.0)
        m.addConstr(A_n @ w == psi_n)
        m.update()

        m.setObjective(sum(w) + 1000 * (sum(psi_p) + sum(psi_n)), GRB.MINIMIZE)

        self.LP = m

    def _run_LP(self, p):
        """
        Return list of indices for conditions for Rule
        """
        self.LP.optimize()
        w_out = []

        for i in range(self.percentiles * p):
            w = int(self.LP.getVarByName('weights[{}]'.format(i)).x)
            if w == 1:
                w_out.append(w)

        return w_out

    def update(self, y_hat, y):
        self.D /= np.exp((2 * y - 1) * y_hat)
        self.D /= np.sum(self.D)

    def fit(self, X, y, T=5):
        """
        Fit ERL model
        """
        # initialize problem
        M = self._make_measurement_matrix(X, y)
        self._make_LP(M, y)
        m = X.shape[0]
        self.D = np.array([1 / m for _ in range(m)])

        for _ in range(T):
            w = self._run_LP(X.shape[1])
            rule = Rule()
            for feat in w:
                rule.conditions.append(self.conditions[feat])

            s_tp, s_fp = rule._get_design_matrices(X, y, self.D)
            s_t = np.sum(self.D[np.where(y == 1)])
            s_f = np.sum(self.D[np.where(y == 0)])

            if np.square(np.sqrt(s_tp) - np.sqrt(s_fp)) \
                    > np.square(np.sqrt(s_t) - np.sqrt(s_f)):
                rule.calc_C_R(s_tp, s_fp, self.D)
                y_hat = rule.C_R * rule.predict(X)
            else:
                rule.calc_C_R(s_t, s_f, self.D)
                y_hat = rule.C_R

            self.rules.append(rule)
            self.update(y_hat, y)
