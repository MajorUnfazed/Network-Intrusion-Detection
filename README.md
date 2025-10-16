# Network Intrusion Detection

Author: **Samuel Thomas**  
Dataset: [KDD Cup 1999](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

---

## Overview
A simple yet effective machine learning model for detecting malicious network connections using the KDD’99 dataset.

This project uses **Logistic Regression** on 15 key numeric features to classify network traffic as either normal or attack.

---

## Steps
1. Download datasets:
   - [Training data (numeric only)](https://drive.google.com/file/d/1Bi5MJwwMJUOfPojXo174xBRV9NeGA_Qp/view?usp=sharing)
   - [Official test set](https://drive.google.com/file/d/1QRinffdxl-aolgiokm6-XJElKGJB1ZSH/view?usp=sharing)
2. Place them in the same folder as `main.py`.
3. Run:
   ```bash
   python main.py
