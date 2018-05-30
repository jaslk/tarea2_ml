from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

#  # # # # # # # # # # # # # # # # # # # 
# MÉTODO DE CLASIFICACIÓN NAIVE BAYES  #
#  # # # # # # # # # # # # # # # # # # # 

model = make_pipeline(TfidfVectorizer(), MultinomialNB())


