from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from TrainTest import trainTestSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def getAccuracyNB():
    X_train, X_test, y_train, y_test = trainTestSplit()

    # model = ('Naive Bayes', GaussianNB())

    model = GaussianNB()

    labels = ["H", "A", "D"]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cMatrix = confusion_matrix(y_test, y_pred, labels=labels)
    cReport = classification_report(y_test, y_pred, labels=labels)
    print(f'Naive Bayes: {accuracy:.2%}')
    print(cMatrix)
    print(cReport)
    print('-' * 50)

    return accuracy


getAccuracyNB()