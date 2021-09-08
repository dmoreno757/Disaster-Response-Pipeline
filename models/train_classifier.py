import sys
import pickle
# import libraries
import matplotlib.pyplot
from sqlalchemy import create_engine
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

# import statements
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np

#pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """
    Load the database file to be used for our training
    
    Parameters:
    database_filepath: The path where the database is located at
    
    returns:
    X: Gathering the Features
    Y: Gathering the labels
    categoryNames: Columns of the data
    """
    engineDB = create_engine('sqlite:///../data/DisasterResponse.db')
    #engineDB = create_engine('sqlite:///DisasterResponse.db')
    dfApp = pd.read_sql('disasterapp', engineDB)
    X = dfApp['message']
    Y = dfApp.iloc[:,4:]
    categoryNames = Y.columns
    
    return X, Y, categoryNames

def tokenize(text): 
    """
    Tokenize the text 
    
    Parameters:
    text: The text that needs to be tokenized
    
    returns:
    cleanTokens: The finished text that has been tokenized
    """
    
    token = word_tokenize(text)
    lemme = WordNetLemmatizer()
    
    cleanTokens = []
    
    for toke in token:
        
        clean = lemme.lemmatize(toke).lower().strip()
        cleanTokens.append(clean)
    
     
    return cleanTokens
    

def build_model():
    """
    Build the pipeline to be used for our training and testing
    
    Parameters:
    None
    
    returns:
    gscv: The ML model to be used as our classifier
    """
    
    pipelineRFC = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    param = {
            'clf__estimator__n_estimators': [10, 50]
    }
    
    gscv = GridSearchCV(pipelineRFC, param_grid=param, verbose=15)
    return gscv
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Test our model and evaluates our results
    
    Parameters:
    model: The model being used for the classification
    X_test: The testing features
    Y_test: The testing labels
    category_names: the category names being used
    
    returns:
    None
    """
    
    
    yPredictorTest = model.predict(X_test)
    
    for idx, col in enumerate(Y_test):
        print(col, classification_report(Y_test[col], yPredictorTest[:, idx]))

def save_model(model, model_filepath):
    """
    Save our Model
    
    Parameters:
    model: The model being used for the classification
    model_filepath: the path where the model will be saved
    
    returns:
    None
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()