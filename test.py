import pandas as pd

train = pd.read_csv('data_aihub/train_sentence.csv')
val = pd.read_csv('data_aihub/val_sentence.csv')

print(len(train))

print(train.columns)

from collections import Counter
print(Counter(train.label))

print(Counter(val.label))