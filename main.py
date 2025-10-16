#Author: Samuel Thomas
#Very simple network intrusion detection prediction using LogisticRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Got this dataset from the official KDD Cup 1999 Data: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
#Download dataset from: https://drive.google.com/file/d/1Bi5MJwwMJUOfPojXo174xBRV9NeGA_Qp/view?usp=drive_link
df = pd.read_csv("kdd_full_numeric.csv")


features = [
    'duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'dst_host_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'is_guest_login'
]

X = df[features].astype(float).values
y = (df['label'] != 'normal').astype(int).values

print("X shape:", X.shape)
print("y shape:", y.shape)   
print("attack ratio:", y.mean())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42, stratify=y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, y_pred))


"""
Below is the official test sample data given by KDD

The reason why the accuracy is only around 80% is because the test set contains new threats that werent in the training set to see how the model would perform when faced with unknown threats.
I know 80% isnt that great , i'll definitely come up with a stronger model that performs better.
"""

#Download this from: https://drive.google.com/file/d/1QRinffdxl-aolgiokm6-XJElKGJB1ZSH/view?usp=sharing
df_corrected = pd.read_csv("corrected_full_42cols.csv")

X_corr = df_corrected[features].astype(float).values
y_corr = (df_corrected["label"] != "normal").astype(int).values

X_corr_scaled = sc.transform(X_corr)
y_corr_pred = logreg.predict(X_corr_scaled)

acc_corr = accuracy_score(y_corr, y_corr_pred)
cm_corr = confusion_matrix(y_corr, y_corr_pred)

print("\n=== OFFICIAL TEST SET ===")
print("Accuracy:", acc_corr)
print("Confusion matrix:\n", cm_corr)
print("\nClassification report:\n", classification_report(y_corr, y_corr_pred))