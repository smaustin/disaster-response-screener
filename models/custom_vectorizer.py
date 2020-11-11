from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer

# Need to lemmatize and use built in tokenizer and case normalizer from CountVectorizer(),
# Create a custom Class inheriting from CountVectorizer()
def lemmatize(tokens):
    """Helper function to lemmatize text during tokenization.
    
    Args: 
    tokens: list. Tokenized list

    Returns:
    lemmed_tokens: list. Tokenized list with lemmatized text
    """   
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    lemmed_tokens = []
    for tok in tokens:
        # lemmatize both nouns and verbs so two passes, 
        lemmed_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok), pos='v')
        lemmed_tokens.append(lemmed_tok)

    return lemmed_tokens

class CustomVectorizer(CountVectorizer):
    """Custom vectorizer that inherits from CountVectorizer.

    Allows for lemmatization to happening during CountVectorizer tokenization 
    and utilizes built-in preprocessing for case etc. """
    
    def build_tokenizer(self):
        """Override build_tokenizer method from CountVectorizer
        
        Returns:
        tokenizer: callable. function to split a string into a sequence of tokens.
        """

        # Invoke build_tokenizer from parent class
        tokenize = super().build_tokenizer()
        return lambda doc: list(lemmatize(tokenize(doc)))

