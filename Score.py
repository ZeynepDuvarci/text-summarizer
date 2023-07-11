import nltk
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import nltk
from stemming.porter2 import stem
import string
import numpy as np
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import gensim
from scipy import spatial
from datasets import load_metric
import evaluate



def p1(sentence):
  tagged_sent = pos_tag(sentence.split())
  propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
  possessives = [word for word in sentence if word.endswith("'s") or word.endswith("s'")]
  result=(len(propernouns)+len(possessives))/len(sentence.split())

  return result

def num_there(s):
  return any(i.isdigit() for i in s)

def p2(sentence):
  count = 0
  list = sentence.split()
  for i in list:
    if num_there(i) == True:
      count = count + 1

  result = count / len(list)
  return result

def p4(title,sentence):
  count=0
  for i in title.split():
    for j in sentence.split():
      if i==j:
        count=count+1
  return count/len(sentence.split())

def remove_punctuation(txt):
  txt_nopunt = "".join([c for c in txt if c not in string.punctuation])
  return txt_nopunt

def nltk_preprocessing(sentence):
  # tokenization
  list = nltk.tokenize.word_tokenize(sentence)
  # stemming
  for i in range(0, len(list)):
    list[i] = stem(list[i])
  # stopword-elimination
  list = [word for word in list if not word in stopwords.words()]
  # punctuation
  for i in range(0, len(list)):
    list[i] = remove_punctuation(list[i])

  string = ""
  for i in list:
    string = string + i + " "

  return string.strip()

def word_embedding(model,sentence1,sentence2):
  #kelime listesi
  words = list(model.index_to_key)

  sentence_list_1 = [w for w in sentence1.split() if w in words]
  sentence_list_2 = [w for w in sentence2.split() if w in words]

  if (len(sentence_list_1)==0 or len(sentence_list_2)==0):
    sim=0
  else:
    sim = model.n_similarity(sentence_list_1, sentence_list_2)

  return round(sim,2)

def get_top_words(sentence_list):
  vectorizer = TfidfVectorizer()
  vectorizer.fit_transform(sentence_list)
  # kelimeler
  feature_names = vectorizer.get_feature_names_out()
  tf_idf_vector = vectorizer.transform(sentence_list)
  coo_matrix = tf_idf_vector.tocoo()
  tuples = zip(coo_matrix.col, coo_matrix.data)
  sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

  # %10 hesaplama
  count = 0
  for i in sentence_list:
    for j in i:
      count = count + 1

  count = int(count / 10)
  sorted_items = sorted_items[:count]

  words = []

  # word index and corresponding tf-idf score
  for idx, score in sorted_items:
    words.append(feature_names[idx])

  return words

def p5(sentence,top_words):
  count=0

  for i in sentence.split():
    for j in top_words:
      if j == i:
        count=count+1

  return count / len(sentence)

def normalization(tuple_list):
  score_list=[]
  for i in tuple_list:
    (index,score)=i
    score_list.append(score)

  #0-1 arası normalize
  norm = []
  diff_arr = max(score_list) - min(score_list)
  for i in score_list:
    temp = (((i - min(score_list)) ) / diff_arr)
    norm.append(temp)

  for i in range(0,len(tuple_list)):
    (index,score)=tuple_list[i]
    tuple_list[i]=(index,norm[i])

  return tuple_list

def rouge(exp,summary):
  rouge=evaluate.load("rouge")
  if len(exp)==len(summary):
    score = rouge.compute(predictions=summary, references=exp)
    return score
  elif len(summary)>len(exp):
    score = rouge.compute(predictions=summary[:len(exp)], references=exp)
    #for i in range(0,len(summary)-len(exp)):
      #exp.append("")
    #score2 = rouge.compute(predictions=summary, references=exp)
    return score
  elif len(exp)>len(summary):
    score = rouge.compute(predictions=summary, references=exp[:len(summary)])
    #for i in range(0, len(exp) - len(summary)):
      #summary.append("")
    #score2 = rouge.compute(predictions=summary, references=exp)
    return score

