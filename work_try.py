import numpy
import os
import csv
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

numpy.random.seed(7)

# csv_file = "train.csv"
# csv_df = pd.read_csv(csv_file, index_col=None, header=0, engine='python')
# print(csv_df.review)
# csv_df = pd.DataFrame(csv_data)
# pf = numpy.array(csv_df)
# print(pf)
pf = numpy.load('pf3.npy')

pf2 = pd.DataFrame(pf)

# print(pf2)
X, y = pf[:, :-1], pf[:, -1]

list_y = []
for i in range(len(y)):
    a = y[i][0]
    list_y.append(a)

print(X)
list_x = []
for i in range(len(X)):
    a = X[i][0]
    list_x.append(a)
    # print(a)


X_train, X_test, y_train, y_test = train_test_split(list_x, list_y, test_size=0.1, random_state=44)


# for i in range(len(y))
# print(X_train)
# print(y_train)
# print(X_test)

max_review_length = 128

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# print(X_train)
# print("??????????????")
# print(X_test)

# begin
embedding_vector_length = 32
model = Sequential()
# model.add(Embedding(10000, embedding_vector_length, input_length=max_review_length))
model.add(Embedding(10000, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("accuracy: %.2f%%" % (scores[1]*100))

model.save("pf_model1.h5")



