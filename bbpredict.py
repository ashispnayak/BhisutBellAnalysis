import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
import pickle
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn.externals import joblib

bbmodel = joblib.load('finalmodel.sav')
vocabulary = pickle.load(open('vocabulary.txt','rb'))
newNoticeName = ['T&P Notice: Yodlee Placement']

#Building the predicting dataframe
notice_ = {'Notice_Name':newNoticeName}
notice_df = pd.DataFrame(notice_,columns=['Notice_Name'])

#Converting to lowercase
notice_lower = [str.lower(sample) for sample in notice_df.Notice_Name]
notice_lower_df = pd.DataFrame(notice_lower,columns=['Notice_Name'])


#Initializing the stemmer
snow = SnowballStemmer('english')

#Building a tokenizer
def token_stem(notice):
	tokens = [word for sent in nltk.tokenize.sent_tokenize(notice) for word in nltk.tokenize.word_tokenize(sent)]
	filtered_tokens = []
	for token in tokens:
		if(re.search('[a-zA-Z]',token)):
			filtered_tokens.append(token)
	notice_stems = [snow.stem(tok) for tok in filtered_tokens]
	return notice_stems


#Building the Word Vectorizer
tfidf_model = TfidfVectorizer(tokenizer=token_stem,stop_words='english',vocabulary=vocabulary,use_idf=True,ngram_range=(1,3))
tfidf_matrix = tfidf_model.fit_transform(notice_lower_df.Notice_Name)

#Predicting the value
final_val = bbmodel.predict(tfidf_matrix)
print(final_val)

