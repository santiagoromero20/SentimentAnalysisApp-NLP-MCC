# Sentiment Analysis on Spotify App Reviews

Spotify it is one of the largest music streaming service providers, with over 422 million monthly active users, including 182 million paying subscribers, as of March 2022. Some of them don't hesitate to share their experience using this application along with the given rating to denote how satisfied they are on Google Play Store Review. The data was collected by scraping Spotify reviews on Google Play Store.

## Motivation

My motivation to start doing this project was to get more knowledge about processes of an NLP project. I have already done a Binary Classification Project, so I wanted to try something harder, a Multi-Class Classification problem. Once I decided this I started looking for datasets and I found the one it is use on Kaggle. Nowadays there are millions of apps being reviewed by users every day, so the idea of automatically analysing a review and knowing which is the sentiment of it, in this particular
case "Positive", "Negative" or "Neutral", it is an amazing real world application of AI. 

Let´s talk a little bit about the project. I wanted to see how different ML models performed for different types of Vectorizations techniques, so on the Notebook you will notice the use of
Logistic Regression and Decision Tree Classifier with input data vectorized with techniques such as Bag of Words, Tf-Idf (built in classes from Scikit-learn) and an trained Word Embedding (I trained my own one with the vocabulary from the dataset).

## Table of Contents

**[1. Sentiment Analysis Notebook](#heading--1)**

  * [1.1. Data Collection](#heading--1-1)
  * [1.2. Data Analysis and Preprocessing](#heading--1-2)
    * [1.2.1. Basic Data Information](#heading--2-1-1)
    * [1.2.2. Handling with Imabalanced Data](#heading--2-1-1)
        * [1.2.2.1. Oversampling: Data Augmentation](#heading--2-1-1)
    * [1.2.3. Text Cleaning](#heading--2-1-1)
    * [1.2.4. WordCloud Visualizations](#heading--2-1-1)
  * [1.3. Scikit-Learn Vectorization Techniques Explanation](#heading--1-2)
    * [1.3.1. Bag Of Words](#heading--2-1-1)
    * [1.3.2. Tf-Idf](#heading--2-1-1)
  *  [1.4. Pipeline Selection](#heading--1-2)
  *  [1.5. Training My Own Word Embedding](#heading--1-2)
     * [1.4.1. Logistic Regression](#heading--2-1-1)
     * [1.4.2. Decision Tree Classifier](#heading--2-1-1)
  
  
On this project, we will code and deploy an API for serving our own machine learning model. For this particular case, it will be for Sentimnt Analysis Classification for the Spotfy App.

Below is the full project structure:

```
├── api
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── config.py
│   ├── utils.py
│   └── routers
│       └── auth.py
│       └── feedback.py
│       
├── test
│   ├── database.py
│   ├── conftest.py
│   ├── test_feedback.py
│   └── test_user.py
│
├── Model
│   ├── SentimentAnalysis.ipynb
│   ├── visualization.py
│   ├── evaluation.py
│   └── text_normalizer.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└──REPORT.md



```

Let's take a quick overview on each module:

- api: It has all the needed code to implement the communication interface between the users and our service. It uses fastapi and PostgreSQL to save user´s feedback and rating.
    - `api/main.py`: Setup and launch our api.
    - `api/database.py`: PostgeSQL Database Initialization.
    - `api/models.py`: Database Tables Declaration with SQLAlchemy.
    - `api/schemas.py`: Database Tables Validation Schemas.
    - `api/routers.py`: Folder which contains the API endpoints. You must implement the following endpoints:
        - *login*: POST method where Spotify User logs in.
        - *create_feedback*: POST method which receives the feedback, rating and a harcoded string which will then be out predicted sentiment. Here we load our feedback on the respective database with its prediction. Very useful for future training.
    - `api/utils.py`: Implements some extra functions used internally by our api.
    - `api/config.py`: It has all the API settings.
  
- test: Contains all the files to perform the testing of the app.
    - `database.py`: PostgeSQL Test Database Initialization.
    - `conftest.py`: Holds the global functions (@fixture) of the routes test.
    - `test_feedback.py`: *create_feedback* tests.
    - `test_user.py`: *login* tests.
  
- model: Contains all the files to perform the Model cleaing, training and evaluation.
    - `visualization.py`: Is the Main Notebook of the proejct where all the code is written. EDA and Model Evaluation is performed here.
    - `visualization.py`: Holds the Visualization Functions.
    - `evaluation.py`: Holds the Evaluation Functions.
    - `text_normalizer.py`: Holds the Text Normalization functions.




