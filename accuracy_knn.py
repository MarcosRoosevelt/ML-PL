
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from TrainTest import trainTestSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


def getAccuracyKNN():
    X_train, X_test, y_train, y_test = trainTestSplit()

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train, y_train_encoded)
    y_pred_encoded = model.predict(X_test)

    y_pred = le.inverse_transform(y_pred_encoded)

    labels = ["H", "A", "D"]
    
    accuracy = accuracy_score(y_test, y_pred)
    cMatrix = confusion_matrix(y_test, y_pred, labels=labels)
    cReport = classification_report(y_test, y_pred, labels=labels)
    print(f'KNN: {accuracy:.2%}')
    print(cMatrix)
    print(cReport)
    print('-' * 50)


    return accuracy


getAccuracyKNN()
