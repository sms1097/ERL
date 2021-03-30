from src.SLIPPER import SLIPPER
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

def main():

    X, y = load_breast_cancer(return_X_y=True)

    clf = SLIPPER()
    clf.fit(X, y)

    preds = clf.predict(X)

    print(accuracy_score(y, preds))

if __name__ == "__main__":
    main()
