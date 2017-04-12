#Code for model
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import re
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
import matplotlib as plt
import numpy as np
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import cosine
from gensim.models import word2vec
from textblob import TextBlob


df_listing = pd.read_csv('listings.csv', encoding = 'utf-8')
df_reviews = pd.read_csv('reviews.csv',encoding = 'utf-8')
df_reviews['date'] = pd.to_datetime(df_reviews['date'])
list_review = pd.merge(df_listing,df_reviews,left_on = 'id', right_on = 'listing_id', how = 'outer')
list_review = list_review.dropna(axis = 0, how = 'any')
lr_set = list_review[:10000]
lr_set = lr_set.reset_index()
lr_set['sentence'] = lr_set['comments'].apply(lambda x : sent_tokenize(x))
temp = deepcopy(lr_set)
temp['string'] = temp.sentence.apply(lambda x: '|'.join(x))
temp_df = temp['string'].str.split('|',expand=True).stack().reset_index()
temp_df.columns=['index_new','throwaway','sentence']
temp_df.drop('throwaway',axis =1, inplace = True)
lr_new = lr_set.merge(temp_df, how='left', left_index=True, right_on='index_new')
lr_new = lr_new.drop('index_new',axis = 1)
lr_new = lr_new.drop('sentence_x',axis = 1)

#Get Sentiment
def p_score(i):
    return TextBlob(i).sentiment[0]

def subjectivity(i):
    return TextBlob(i).sentiment[1]
lr_new['polarity'] = lr_new['sentence_y'].apply(p_score)
lr_new['subjectivity'] = lr_new['sentence_y'].apply(subjectivity)

#Get Topics
stoplist = stopwords.words('english')
def tokenize(line):
    text = []
    words = line.lower().split()
    for word in words:
        word = re.sub(r'[$%&\.\?!,;\'-:0-9\"]',' ',word)
        word = word.rstrip().lstrip()
        
        if word not in stoplist:
            text.append(word)
    return ' '.join(text) 

lr_new['tokenize'] = lr_new['sentence_y'].apply(tokenize)

n_top_words = 20
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" , ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


tfidf = TfidfVectorizer(ngram_range= (1,1))
tfidf_trans = tfidf.fit_transform(lr_new['tokenize'])

nmf = NMF(n_components=20, random_state=1,alpha=.1, l1_ratio=.5)
nmf_ftrans = nmf.fit_transform(tfidf_trans)
feature_names = tfidf.get_feature_names()
# print_top_words(nmf, feature_names, n_top_words)

def sumlist(x):
    for i in range(20,len(x)):
        x[i%20]+=x[i]
    return x[:20]

list_id = []
match = []
def get_listing(words):
    list_id = []
    match = []
    tfidf_words = tfidf.transform(words)
    test=nmf.transform(tfidf_words)
    lr_new['features'] = nmf_ftrans.tolist()
    listings = lr_new.groupby('id_x').features.sum().apply(sumlist).index
    listing_features = lr_new.groupby('id_x').features.sum().apply(sumlist).tolist()
    pair_wise = [(cosine_similarity(test, np.array(v).reshape(1,-1))[0][0],j) for j,v in enumerate(listing_features)]
    top_listings = sorted(pair_wise,reverse=True)[0:10]
    for i in top_listings:
        list_id.append(i[1])
    
    for i in list_id:
        match.append(lr_set.loc[lr_set['index'] == i,['name','host_name','neighbourhood_group',
                                                  'comments']])
    return match[:5]





import flask

# Initialize the app
app = flask.Flask(__name__)

#loads the page
@app.route("/")
def viz_page():
    with open("airbnb.html", 'r') as viz_file:
        return viz_file.read()
    
#listens
@app.route("/gof", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this url,
    Read the grid from the json, update and send it back
    """
    #html "posts" a request and python gets the json  from that request 
    data = flask.request.json
    a = data['grid']
    print(a)
    #d = [c[0] for c in AliIsAwesome.most_similar(a)]
    d = get_listing(a)
    d = [dd.values.tolist()[0] for dd in d]
    return flask.jsonify({'words': d})

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0', port=5002, debug = True)

