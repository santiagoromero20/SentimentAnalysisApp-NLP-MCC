# Sentiment Analysis on Spotify App Reviews

Spotify it is one of the largest music streaming service providers, with over 422 million monthly active users, including 182 million paying subscribers, as of March 2022. Some of them don't hesitate to share their experience using this application along with the given rating to denote how satisfied they are on Google Play Store Review. The data was collected by scraping Spotify reviews on Google Play Store.

## Motivation

My motivation to start doing this project was to get more knowledge about processes of an NLP project. I have already done a Binary Classification Project, so I wanted to try something harder, a Multi-Class Classification problem. Once I decided this I started looking for datasets and I found the one it is use on Kaggle. Nowadays there are millions of apps being reviewed by users every day, so the idea of automatically analysing a review and knowing which is the sentiment of it, in this particular
case "Positive", "Negative" or "Neutral", it is an amazing real world application of AI. 

Let´s talk a little bit about the project. I wanted to see how different ML models performed for different types of Vectorizations techniques, so on the Notebook you will notice the use of
Logistic Regression and Decision Tree Classifier with input data vectorized with techniques such as Bag of Words, Tf-Idf (built in classes from Scikit-learn) and an trained Word Embedding (I trained my own one with the vocabulary from the dataset).

## Table of Contents

**[1. EDA Notebook](#heading--1)**

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
├── spotify_pipe.pkl
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── REPORT.md



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

- `spotify_pipe.pkl`: Is the best performer scikit-learn pipeline, is the one in charge of all the steps of a typical ML Pipeline, starting from Vectorizing up until making the prediction. It is use on *create_feedback* route. You can read a little bit more about it on the *REPORT.md* file.


## Installation

You can pull the image from DockerHub, to do so you must have an account. If you don´t, you can clone my project into your local repo and run this command:

      docker-compose -f docker-compose-prod.yml up -d

By doing this you will be building the image and start running your container on the same step, it isn´t mandatory to do it in this way as there are others, it´s just a recommendation. On the Dockerfile, you can see the use of two services (the app itself and the database) which logically are started by different images. The App image is declare and "build" it on the Dockerfile and the Postgresql is the official one from DockerHub. 

Be aware that if you change the places from files of folders or anything like that you may be probably harming the full functionallity of the API for different reasons, and that you should declare your own environment variables for initializing your database.

Once the Container is running you can go to your **localhost/docs** url and try for yourself all the routes.



