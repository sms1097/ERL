from src.SLIPPER import SLIPPER
from sklearn.datasets import load_breast_cancer

def main():

    X, y = load_breast_cancer(return_X_y=True)

    clf = SLIPPER()
    clf.fit(X, y)


if __name__ == "__main__":
    main()
