import pandas as pd
import string
import csv
import nltk
import numpy as np
from gensim import corpora
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint  # pretty-printer

f_data = pd.read_csv('train.csv', index_col=None, header=0, engine='python')

print(f_data.describe())

record_num = int(f_data.describe().iloc[0, 0])

# 查看第一行所有数据
print(f_data.iloc[0, :])
documents = []
# list.append(f_data.iloc[0, :]['review'])
# list.append(f_data.iloc[1, :]['review'])
# print(list)

for i in range(record_num):
    record = f_data.iloc[i, :]
    documents.append(record['review'])
    # message = record['review'].split()


# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

# pprint(texts)


dictionary = corpora.Dictionary(texts)
dictionary.save('pf.dict')  # store the dictionary, for future reference

# 🐎的7000多维向量
print(dictionary)
# 输出字典mapping
print(dictionary.token2id)

dictionary.save_as_text("pf.txt")

# tryit = corpora.Dictionary('dictionary')
# print(tryit)


new_doc = "Affia ki"
new_vec = dictionary.doc2bow(new_doc.lower().split())
new_vec2 = dictionary.doc2idx(new_doc.lower().split())
for j in range(len(new_vec2)):
    if new_vec2[j] == -1:
        new_vec2[j] = 0
print(new_vec)
print(new_vec2)


msg = []
critic = []

for i in range(record_num):
    record = dictionary.doc2idx(f_data.iloc[i, :]['review'].lower().split())
    msg.append(record)
    if f_data.iloc[i, :]['label'] == 'Positive':
        critic.append(1)
    else:
        critic.append(0)

numpy_array_msg = np.array(msg)

numpy_array_critic = np.array(critic)
need = np.array([numpy_array_msg, numpy_array_critic])

# print(need)

np.save('pf.npy', need)

msg2 = []

for i in range(record_num):
    record = dictionary.doc2idx(f_data.iloc[i, :]['review'].lower().split())
    if f_data.iloc[i, :]['label'] == 'Positive':
        record.append(1)
    else:
        record.append(0)
    msg2.append(np.array(record))

numpy_array_msg2 = np.array(msg2)
need2 = np.array(msg2)

# print(need2)

np.save('pf2.npy', need2)


need3 = []
for i in range(record_num):
    critic = []
    df = []
    record = dictionary.doc2idx(f_data.iloc[i, :]['review'].lower().split())
    for j in range(len(record)):
        if record[j] == -1:
            record[j] = 0
    df.append(record)
    if f_data.iloc[i, :]['label'] == 'Positive':
        critic.append(1)
        df.append(critic)
    else:
        critic.append(0)
        df.append(critic)
    need3.append(df)

# print(np.array(need3))

np.save('pf3.npy', np.array(need3))

# 创建词袋模型
# vectorizer = CountVectorizer()
#
# # 使用给定的语料库来训练该模型
# vectorizer.fit_transform(documents)
# # 词表中的每一维相当于一个特征
# print(vectorizer.get_feature_names())
#
# # 查看词表中单词对应的索引
# print(vectorizer.vocabulary_)
#
# get_out = "dictionary"
#
#
# test = ['bhi tou a app']
# result = vectorizer.transform(test)
# # a 和 apple不存在词表中，会直接忽略掉
# print(result.toarray())
