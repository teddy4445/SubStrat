import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def test():
    df = pd.read_csv(r"C:\Users\lazeb\Desktop\stability_feature_selection_dataset_summary\big_data\dataset_26_poker.data")
    print("load data")
    x, y = df.drop([list(df)[-1]], axis=1), df[list(df)[-1]]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    m = RandomForestClassifier(max_depth=4, criterion="entropy")
    #m = SVC(gamma="auto")
    m.fit(X_train, y_train)
    print(m.score(X_test, y_test))

if __name__ == '__main__':
    test()
