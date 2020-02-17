import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn import metrics


# mushrooms_df.head()


def prepared_df():
    # load the data
    mushrooms_df = pd.read_csv('mushrooms.csv').sample(frac=1)
    for class_label in mushrooms_df.columns:
        mushrooms_df[class_label] = LabelEncoder().fit(mushrooms_df[class_label]).transform(mushrooms_df[class_label])

    return mushrooms_df


def calculate_adaboost_model_accuracy(mushrooms_df):
    X = mushrooms_df.drop(['class'], axis=1)
    Y = mushrooms_df['class']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    model = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    AdaBoost = AdaBoostClassifier(base_estimator=model, n_estimators=26, learning_rate=1)
    boost_model = AdaBoost.fit(X_train, Y_train)

    Y_predict = boost_model.predict(X_test)

    predictions = metrics.accuracy_score(Y_test, Y_predict)

    print("Accuracy is ", predictions * 100)


def main():
    mushrooms_df = prepared_df()
    calculate_adaboost_model_accuracy(mushrooms_df)


if __name__ == '__main__':
    main()