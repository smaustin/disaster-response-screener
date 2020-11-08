# Disaster Response Pipeline Project
disaster-response-screener is a Flask web application where an emergency worker can input a new message and get classification results in one of 36 categories. The web app also displays visualizations of the model training data. This application could be used to categorize these events in order to send the messages to an appropriate disaster relief agency.

## Base Dependencies:
This project does require Python 3 along with the following dependencies:
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

## Instructions:
To run disaster-response-screener, use the following command from the app directory of the project:

```
python run.py
```
In a new web browser window type in the following:
```
http://0.0.0.0:3001/
```
### ETL and ML Option
In addition to simply running the web application, there is an option to run the operations that setup the database and model used by the web application as described below. 

The first step of this Python project will perform Extract, Transform and Load (ETL) operations using a file containing messages and a file containing categories.  This operation will read the datasets, clean the data, combine the datasets and then store the result in a SQLite database. The next step in the project will run a machine learning (ML) pipeline on the created database. This ML pipeline uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a multilabel classification model. The final step will run the Flask application mentioned above using the model created by the ML pipeline. 

1. Run the following commands in the project's root directory to set up the database.

    ```
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```
2. Run the following commands in the project's root directory to set up the model.

    ```
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```

3. Follow the instructions above to run the web application.
