# from src.SLIPPER import SLIPPER
from src.ERL import ERL
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

MANAGEMENT = 'Light'
DISCOUNT = 'DR5'


def base_load(data, management, discount, spatial=False):

    rates = ["NoDR", "DR1", "DR3", "DR5"]

    if discount not in rates:
        raise ValueError("Invalid Discount Rate!")

    salvage = data[
        (data['TimeStep'] == 40) &
        (data['Treatment'] == management) &
        (data['Salvage'] == 'Salvage')
    ]

    salvage = salvage.set_index("StandID")
    salvage = salvage.fillna(salvage.mean())

    no_salvage = data[
        (data['TimeStep'] == 40) &
        (data['Treatment'] == management) &
        (data['Salvage'] == 'NoSalvage')
    ]

    no_salvage = no_salvage.set_index("StandID")
    no_salvage = no_salvage.fillna(no_salvage.mean())

    data = salvage.copy()
    data['diff'] = data[discount]
    data['diff'] -= no_salvage[discount]

    return data, salvage, no_salvage


def load_data_class(data, management, discount="DR5", spatial=False):
    rates = ["NoDR", "DR1", "DR3", "DR5"]

    data, salvage, no_salvage = base_load(data, management, discount, spatial)
    data['Voucher'] = (data['diff'] > 0)

    rates.remove(discount)
    data = data.drop(rates, axis=1)

    return data, salvage, no_salvage


def get_strategy(data, target, management, discount):
    # Get Optimal Strategy
    strategy = target.rename('strategy')

    salvage = pd.merge(data, strategy, on="StandID")
    salvage_strategy = salvage[
        (salvage['strategy'] == 1) &  # flake8 doesn't like True here
        (salvage['Salvage'] == 'NoSalvage') &
        (salvage['TimeStep'] == 40) &
        (salvage['Treatment'] == management)
    ]
    salvage_strategy = salvage_strategy[discount]

    no_salvage = pd.merge(data, strategy, on="StandID")
    no_salvage_strategy = no_salvage[
        (no_salvage['strategy'] == 0) &  # flake8 doesn't like False here
        (no_salvage['Salvage'] == 'Salvage') &
        (no_salvage['TimeStep'] == 40) &
        (no_salvage['Treatment'] == management)
    ]
    no_salvage_strategy = no_salvage_strategy[discount]

    # Make sure we don't duplicate
    assert target.shape[0] == salvage_strategy.shape[0] + \
        no_salvage_strategy.shape[0]

    # Really make sure we don't duplicate
    a = salvage_strategy.index.tolist()
    b = no_salvage_strategy.index.tolist()
    assert len(set(a).intersection(set(b))) == 0

    outcome = (salvage_strategy.sum() + no_salvage_strategy.sum()) \
        / target.shape[0]

    return outcome


def main():
    all_data = pd.read_csv('data.csv')
    data, salvage, no_salvage = load_data_class(all_data, MANAGEMENT,
                                                DISCOUNT)

    # emissions = [salvage[DISCOUNT], no_salvage[DISCOUNT]]
    # headers = ['salvage', 'no_salvage']
    # optimal_df = pd.concat(emissions, axis=1, keys=headers)
    # optimal_df['positive'] = optimal_df[['salvage', 'no_salvage']].min(axis=1)
    # optimal_df['negative'] = optimal_df[['salvage', 'no_salvage']].max(axis=1)

    X = data.drop(['Voucher', 'Treatment', DISCOUNT, 'diff', 'SiteInd',
                   'Salvage', 'TimeStep'], axis=1)
    y = data['Voucher']

    index = y.index.to_list()

    X, y = X.to_numpy(), y.to_numpy()
    X_train, X_test, y_train, y_test, _, test_index = train_test_split(
        X, y, index, test_size=0.2, random_state=1)

    _, _, E_p, y_no_salvage = train_test_split(
        X, salvage[DISCOUNT], test_size=0.2, random_state=1)

    _, _, E_n, y_salvage = train_test_split(
        X, no_salvage[DISCOUNT], test_size=0.2, random_state=1)

    E_p, E_n = E_p.to_numpy(), E_n.to_numpy()

    clf = ERL()
    clf.fit(X_train, y_train, E_p, E_n)

    preds = clf.predict(X_test)

    preds = pd.Series(preds)
    preds.index = test_index
    preds = preds.rename_axis('StandID')

    y_test = pd.Series(y_test)
    y_test.index = test_index
    y_test = y_test.rename_axis('StandID')

    print(np.unique(preds, return_counts=True))
    print(np.unique(y_test, return_counts=True))

    print('accuracy: {}'.format(accuracy_score(y_test, preds)))
    print('precision: {}'.format(precision_score(y_test, preds)))
    print('recall: {} \n'.format(recall_score(y_test, preds)))

    print('model strategey: {}'.format(
        get_strategy(all_data, preds, MANAGEMENT, DISCOUNT)))
    print('optimal strategey: {}'.format(
        get_strategy(all_data, y_test, MANAGEMENT, DISCOUNT)))

    print('salvage strategy: {}'.format(np.mean(y_salvage)))
    print('no salvage strategy: {}'.format(np.mean(y_no_salvage)) + '\n')

    for rule in clf.rules:
        print(rule)


if __name__ == "__main__":
    main()
