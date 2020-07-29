# Serve model as a flask application
from flask import Flask, request, jsonify, render_template
import pickle
from flask import Flask, request
from flask_cors import CORS
import nltk
import string
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from sklearn.feature_extraction import text





model = None
app = Flask(__name__)

CORS(app)

# def load_model():
#     global model
#     # model variable refers to the global variable
#     with open('finalized_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     global tv


@app.route('/', methods=['GET'])
def classify():
    # if request.method == 'GET':
    return render_template('home_page.html')

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

lem = nltk.WordNetLemmatizer()

single_letters = set(string.ascii_lowercase)
nltk_stop_words = set(nltk.corpus.stopwords.words('english')).union(single_letters)
stopwords = set(text.ENGLISH_STOP_WORDS.union(nltk_stop_words))

error = ['far', 'make', 'need', 'sha', 'wo']
stopwords = set(stopwords.union(error))

def tokenize_and_lemmatize(para):
            tokens = [word for word in nltk.word_tokenize(para)]
            
            #remove non alphabetical tokens
            filtered_tokens = []
            for token in tokens:
                if token.isalpha():
                    filtered_tokens.append(token)
        
            lemmatized_tokens = []        
        
            for (item,pos_tag) in nltk.pos_tag(filtered_tokens):
                lemmatized_token = lem.lemmatize(item, get_wordnet_pos(pos_tag))
                lemmatized_tokens.append(lemmatized_token)
                                    
            #return filtered_tokens
            return lemmatized_tokens 


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json

        print(data)
        


        # with open('new_finalized_model.pkl', 'rb') as f:
        #     model = pickle.load(f)
        with open('tv.pkl', 'rb') as f:
            tv = pickle.load(f)
        with open('finalized_model.pkl', 'rb') as f:
            model = pickle.load(f)


        processed_data = tv.transform([data['lyrics']])
        prediction = model.predict(processed_data)  # runs globally loaded model on the data
    
        
        if prediction.tolist() == [0]:
            print('Male')
            return jsonify({"message": "Male"}), 201
        if prediction.tolist() == [1]:
            print('Female')
            return jsonify({"message": "Female"}), 201
    return jsonify({"message": "Error"}), 400
    


if __name__ == '__main__':
    # load_model()  # load model at the beginning once only
    app.run(port=5000, debug=True)