from keras.models import load_model
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

data = pd.read_csv('test_data2.csv', index_col=None, header=0, engine='python')

print(data.describe())
record_num = int(data.describe().iloc[0, 0])

documents = []

for i in range(record_num):
    record = data.iloc[i, :]
    documents.append(record['review'])

print(documents)

# f = open('pf.txt', 'r')
# dictionary = {}
# for line in f.readlines():
#     line = line.strip()
#     if not len(line):
#         continue
#     dictionary[line.split(':')[0]] = line.split(':')[1]
# f.close()

dictionary = {}
with open("pf.dict", "rb") as f:
    dictionary = pickle.load(f)

max_review_length = 128

list_data = []
for document in documents:
    new_vec2 = dictionary.doc2idx(document.lower().split())
    for j in range(len(new_vec2)):
        if new_vec2[j] == -1:
            new_vec2[j] = 0
    list_data.append(np.array(new_vec2))

list_data = np.array(list_data)

X = sequence.pad_sequences(list_data, maxlen=max_review_length)
print(X)


model = load_model('pf_model1.h5')
print(X.shape)
# predict = model.predict_classes(X)
predict = model.predict(X)
print(predict)
data['Pred'] = predict
result = data[['ID', 'Pred']]
result.to_csv('result2.csv', index=False)
# for l in X:
#     print(l.shape)
#     predict = model.predict_classes(l)
#     print(predict)

# print(dictionary)
