from utils.config import *
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def split_data():
    data_dict = pickle.load(open(DATA_PICKLE_FN, 'rb'))
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    return (X_train, X_test, y_train, y_test)


def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print('Accuracy:', score)


def save_model(model):
    f = open(MODEL_FN, 'wb')
    pickle.dump({'model': model}, f)
    f.close()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)