from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer

# Need to lemmatize and use built in tokenizer and case normalizer from CountVectorizer(),
# Create a custom Class inheriting from CountVectorizer()
def lemmatize(tokens):
    """Helper function to lemmatize text during tokenization."""   
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize both nouns and verbs so two passes, 
        clean_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok), pos='v')
        clean_tokens.append(clean_tok)

    return clean_tokens

class CustomVectorizer(CountVectorizer):
    """Custom vectorizer that inherits from CountVectorizer.
    Allows for lemmatization to happening during CountVectorizer tokenization 
    and utilizes built-in preprocessing for case etc."""
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(lemmatize(tokenize(doc)))

