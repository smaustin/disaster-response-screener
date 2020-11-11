# Disaster Response Pipeline Project
disaster-response-screener is a Flask web application where an emergency worker can input a new message, received directly of through social media, and have it classified into any number of 36 categories. In a disaster scenario, where many messages are being received, this application could be used to quickly provide an initial categorization of messages.  These results could then be used to efficiently route messages to the appropriate disaster relief agency. The web application also displays visualizations of the data used to train the model. 

## Base Dependencies:
This project does require Python 3.6 or above, along with the following dependencies:
```
numpy
pandas
sqlalchemy
pickle
scikit-learn
nltk
flask
plotly
joblib
```
## Data and File Overview
The data used to train and test the model in this project was provided by [Figure Eight](https://www.figure-eight.com/). The data consists of thousands of actual messages that were sent during natural disasters and then later categorized by Figure Eight. This data was provided in two CSV files, one containing the messages (disaster_messages.csv) and one containing the category classifications (disaster_categories.csv).

An outline and summary of the files contained in this project are provided below.

```
├── README.md
├── app
│   ├── run.py # Flask file that runs app
│   └── templates
│       ├── go.html # classification results page of web app
│       └── master.html # main page of web app
├── data
│   ├── DisasterResponse.db # database of cleaned disaster data
│   ├── disaster_categories.csv # raw disaster category data
│   ├── disaster_messages.csv # raw disaster message data
│   └── process_data.py # file to clean data and generate database file
└── models
    ├── custom_vectorizer.py # custom vectorizer used in train_classifier
    └── train_classifier.py # file to train, evaluate and save model
```

## Instructions:
The model used to evaluate new messages is too large to store in a repository, and therefore must be created and saved to disk before running the web application. In addition to creating the model and running the web application, there is also an option to run the operations that setup the database. These procedures are described below. 

The first step of this Python project is optional and will perform Extract, Transform and Load (ETL) operations on files containing message and category data.  This operation will read the datasets, clean the data, combine the datasets and then store the result in a SQLite database. The next step in the project will run a machine learning (ML) pipeline on the created database. This ML pipeline uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to create a multilabel classification model and save it as a pickle file. The final step will run the Flask based web application using the model created in the ML pipeline step. 

1. **ETL Pipeline** (optional): Run the following commands in the project's root directory to run the ETL pipeline and set up the database.

    ```
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```
2. **ML Pipeline**: Run the following commands in the project's root directory to set up the model and save it to a pickle file for use by the web app.

    ```
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```

3. **Web App**: To run disaster-response-screener, use the following command from the app directory of the project:

    ```
    python run.py
    ```
    In a new web browser window type in the following:
    ```
    http://0.0.0.0:3001/
    ```
