# from src.SLIPPER import SLIPPER
from src.ERL import ERL
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def base_load(management, discount, spatial=False):

    rates = ["NoDR", "DR1", "DR3", "DR5"]

    if discount not in rates:
        raise ValueError("Invalid Discount Rate!")

    data = pd.read_csv("data.csv")

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
    data[discount] -= no_salvage[discount]

    return data


def load_data_class(management, discount="DR5", spatial=False):
    rates = ["NoDR", "DR1", "DR3", "DR5"]

    data = base_load(management, discount, spatial)
    data['Voucher'] = (data[discount] > 0)

    rates.remove(discount)
    data = data.drop(rates, axis=1)

    return data


def main():
    data = load_data_class('Light', 'DR5')

    X = data.drop(['Voucher', 'Treatment', 'DR5', 'SiteInd', 'Salvage',
                   'TimeStep'], axis=1)
    y = data['Voucher']
    # X, y = load_breast_cancer(return_X_y=True)

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1)

    clf = ERL()
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print(accuracy_score(y_test, preds))

    print(np.unique(y_test, return_counts=True))
    print(np.unique(preds, return_counts=True))

    for rule in clf.rules:
        print(rule)

if __name__ == "__main__":
    main()
