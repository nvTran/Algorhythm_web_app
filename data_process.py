import pandas as pd
import re
import numpy as np
import nltk
import string
from collections import Counter
from math import sqrt
import csv


from sklearn.feature_extraction import text

# data = open("Test text document.txt", "r")
# trained_df = open("saved_df.csv","r")
# # print(data.read())



def process_data(raw_lyrics_data):
    data= str(raw_lyrics_data)
    def remove_name_assignment(str):
        occurences = re.findall(r'\[.*?\]',str)
        for item in occurences:
            str = str.replace(item,'')
        return str

    informal_contractions = {
        'wanna':'want to', 'gonna':'going to','tryna':'trying to',
        'gotta':'got to', 'sorta':'sort of','outta':'out of', 'alotta':'lot of','lotta':'lot of',
        'oughta':'ought to','usta':'used to','supposeta':'supposed to','mighta':'might have','musta':'must have',
        'kinda':'kind of', 'needa':'need to','cuppa':'cup of',
        'gimme':'give me','lemme':'let me',
        'hasta':'has to','hafta':'have to',
        'gotcha':'got you','whatcha':'what you','betcha':'bet you','dontcha':'don\'t you',
        'didntcha':'didn\'t you','wonntcha':'won\'t you',
        'c\'mon':'come on','s\'more':'some more',
        'dunno':'don\'t know',
        'shoulda':'should have', 'shouldna':'should not have',
        'coulda':'could have', 'couldna':'could not have',
        'woulda':'would have', 'wouldna':'would not have',
        'innit':'isn\'t it', 'ev\'ry':'every'
    }

    def get_wordnet_pos(treebank_tag):

        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return nltk.corpus.wordnet.NOUN

    english_dict = set(nltk.corpus.words.words())

    lem = nltk.WordNetLemmatizer()
        
    single_letters = set(string.ascii_lowercase)
    nltk_stop_words = set(nltk.corpus.stopwords.words('english')).union(single_letters)
    stopwords = text.ENGLISH_STOP_WORDS.union(nltk_stop_words)
    def tokenize_and_lemmatize(doc):
        tokens = [word for sent in doc.split('\n') for word in nltk.word_tokenize(sent)]
        
        #remove non alphabetical tokens
        filtered_tokens = []
        for token in tokens:
            if token.isalpha():
                filtered_tokens.append(token)
    
        lemmatized_tokens = []        
    
        for (item,pos_tag) in nltk.pos_tag(filtered_tokens):
            lemmatized_token = lem.lemmatize(item, get_wordnet_pos(pos_tag))
            if lemmatized_token in english_dict:
                lemmatized_tokens.append(lemmatized_token)
                                
        #return filtered_tokens
        return lemmatized_tokens 

    ourData = {} 
    with open('test.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ourData[row[0]]= float(row[1])


    def input_preprocessing(data):
        data = data.lower()
        data = data.replace('<i>','').replace('</i>','').replace('in\'','ing').replace('aingt','ain').replace('\'',' ')
        data = remove_name_assignment(data)
        for key,value in informal_contractions.items():
            data = data.replace(key,value)
            
        tokens = tokenize_and_lemmatize(data)
        
        # filtered include non stop words 
        nostopwords = []
        for token in tokens:
            if token not in stopwords:
                nostopwords.append(token)
                
        # list of only words belonged in top 200
        keywords = []
        for token in nostopwords:
            if token in ourData.keys():
                keywords.append(token)
                
        result_df = pd.DataFrame(0, range(1), columns = ourData.keys())
        
                
        # tf / term frequency / bag of words:
        bow = dict(Counter(keywords))
        
        # NOT YET normalized tfidf
        normalizer = 0
        for key in bow.keys():
            result_df[key] = bow[key]*ourData[key]
            normalizer += result_df[key]**2
        normalizer = sqrt(normalizer)
        
        # normalized tfidf
        for key in bow.keys():
            result_df[key] = result_df[key]/normalizer     
        return result_df
    
    
    return input_preprocessing(data)

def process_data_pickle(raw_lyrics_data):
    infile = open('tfidf','rb')
    new_dict = pickle.load(infile)

