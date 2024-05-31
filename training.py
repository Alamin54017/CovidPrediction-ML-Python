import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

from sklearn.metrics import accuracy_score


def data_split(data, ratio):
    shuffled = np.random.permutation(len(data))
    test_size = int(len(data) * ratio)
    test_indices = shuffled[:test_size]
    train_indices = shuffled[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
if __name__ == "__main__":
    df=pd.read_csv("data.csv")
    train, test = data_split(df,0.2)

    X_train = train[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreathe', 'abroadHist']].to_numpy()
    X_test = test[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreathe', 'abroadHist']].to_numpy()

    Y_train = train[['infectionProb']].to_numpy().reshape(2400, )
    Y_test = test[['infectionProb']].to_numpy().reshape(599, )

    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    y_pred_logreg = clf.predict(X_test)
    print('Logistic Regression Train accuracy %s' % clf.score(X_train, Y_train))
    print('Logistic Regression Test accuracy %s' % accuracy_score(y_pred_logreg, Y_test))
    file = open('model.pkl', 'wb')
    

    pickle.dump(clf, file)


    file.close()
    

