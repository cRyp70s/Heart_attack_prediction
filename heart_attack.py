from typing import List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier


def train_classifier(data: str) -> RandomForestClassifier:
    df = pd.read_csv(data)
    X = df.drop(columns=["DEATH_EVENT"])
    y = df["DEATH_EVENT"]
    clf = RandomForestClassifier(max_depth=300, n_estimators=100)
    clf.fit(X, y)
    return clf

def predict(clf: RandomForestClassifier, data: List[List]) -> List:
    probabilities = clf.predict_proba(data)
    predictions = []
    for prob in probabilities:
        if prob[0] > prob[1]:
            predictions.append([False, prob[0]])
        else:
            predictions.append([True, prob[0]])
    return predictions
