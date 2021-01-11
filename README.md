# Disaster-Response-Project-Schoolwork
This is a Machine Learning(ML) powered web app for classifying disaster response messages. It contains a web interface where a user can input a message and classify a message into several categories thereby reducing the time taken to manually label messages into categories; The model is trained specifically to classify disaster related messages into categories such as clothing, food, death, hospitals etc

## Table of Contents
1. [Project Motivation](#motivation)
2. [Installations](#installations)
3. [File Overview](#overview)
4. [Instructions](#instructions)
5. [Acknowledgements](#acknowledgements)

## <a id="motivation"/> Project Motivation
The motivation of this project was to build a machine learning based web app to help categorize disaster related messages using python's Machine Learning library scikit-learn and natural languae processing library nltk. I was also looking to put in practice on building ETL pipelines and in particular transforming and cleaning data to be ready to be used in training a model.

## <a id="installations"/> Installations
This project is mainly written in Python3 and the frontend utilizes HTML. The project requires the following packages to be installed:
- pandas
- numpy
- sys
- sqlalchemy
- re
- pickle
- nltk
- sklearn
- plotly
- json
- joblib
- flask

##  <a id="overview"/> File Overview
Files and folders included in this project include:
- app: This folder contains:
    - templates folder : the HTML files required to render the web app
    - `run.py`: a python file which starts the app
- data: This folder contains:
    - disaster_messages.csv file : sample messages datasets to be used for model training
    - disaster_categories.csv file: sample categories of the messages to be used for model training
    - process_data.py : a python file that reads in the messages and categories csv files, merges them together, cleans/transforms the merged dataframe  and then creates a SQLite database with the cleaned dataframe as a table
- models: This folder contains
 - trian_classifier.py : a python file that loads the SQLite table produced by process.py and uses this data to train a ML model as well as evaluate the performance of the trianed model. The trained model is then saved as a pickle file to be used for predicting categories of new messages
 NB: The pickle file is omitted from this repo because of its large file size but upon training the model locally  the pickle file is ave under this folder.

## <a id="instructions"/> Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python3 models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python3 run.py`

3. Go to http://0.0.0.0:3001/

## <a id="acknowledgements"/> Licenses and acknowledgements
This project was worked on as part of the Data Engineering module of the Udacity Data Scientist Nanodegree. The datasets are from Figure Eight who have collaborated with Udacity to provide data to be used to complete this ML project. Code templates were also provided by [Udacity](https://www.udacity.com/) .