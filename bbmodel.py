import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn.externals import joblib
import pickle

#Loading the data
notices = pd.read_csv('final_notices.csv')
notices.Label = notices.Label.astype('int64')
notices_label = notices.Label
notices_names = notices.drop('Label',axis=1)


#Converting into lowercase
notices_lower = [str.lower(notice) for notice in notices_names.Notice_Name]
notices_lower_df = pd.DataFrame(notices_lower,columns=['Notice_Name'])



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

#Building the vocabulary
totalvocab = []
for notice in notices_lower_df.Notice_Name:
    allwords = token_stem(notice)
    totalvocab.extend(allwords)

finalvocab = list(set(totalvocab))

#Saving the vocabulary in disk
pickle.dump(finalvocab,open('vocabulary.txt','wb'))

    
#Building the Word Vectorizer
tfidf_model = TfidfVectorizer(tokenizer=token_stem,stop_words='english',vocabulary=finalvocab,use_idf=True,ngram_range=(1,3))
tfidf_matrix = tfidf_model.fit_transform(notices_lower_df.Notice_Name)

#Building the clustering the model
lr_model = LogisticRegression()
lr_model.fit(tfidf_matrix,notices_label)
joblib.dump(lr_model,'finalmodel.sav')
