# Sentiment Analysis on Spotify App Reviews

Overview:
Spotify it is one of the largest music streaming service providers, with over 422 million monthly active users, including 182 million paying subscribers, as of March 2022. Some of them don't hesitate to share their experience using this application along with the given rating to denote how satisfied they are on Google Play Store Review. The data was collected by scraping Spotify reviews on Google Play Store.

# Project Motivation and Description

My motivation to start doing this project was to get more knowledge about processes of an NLP project. I have already done a Binary Classification problem on Sentiment Analysis (https://github.com/santiagoromero20/SentimentAnalysisMovies), so I wanted to try something harder, a Multi-Class Classification problem. Once I decided this I started looking for dataset and I found the one it is use on Kaggle. Nowadays there are more millions of apps being reviewed by users every day, so the idea of automatically analysing a review and knowing which is the sentiment of it, in this particular
case positive, negative or neutral, it is an amazing real world application of AI. Just doing quick maths, imagine the employee of Spotify in charge of doing this job, let´s suposse it takes you 1 minute
to read a Review, label it with the corresponding sentiment and then uploaded to a generic Database. In one month of work (8hs.20days=9600mins), he could have loaded just around of 10,000 reviews. And 
with an automated ML model this could be done review after review without the need of any repetitive human labour.

Let´s talk a little bit about the project itself. I wanted to see how different ML models performed for different types of Vectorizations techniques, so on the Notebook you will notice the use of
Logistic Regression, Random Forest, KNN, SVM and Naive Bayes Classifiers with input data vectorized with techniques such as Bag of Words, Tf-Idf and Word Embedding (I trained my own word 
embedding with the vocabulary from the dataset).
## Installation

You can use `Docker` to easily install all the needed packages and libraries:

```bash
$ docker build -t s07_project .
```

### Run Docker

```bash
$ docker run --rm --net host -it \
    -v $(pwd):/home/app/src \
    s07_project \
    bash
```

## Run Project

It doesn't matter if you are inside or outside a Docker container, in order to execute the project you need to launch a Jupyter notebook server running:

```bash
$ jupyter notebook
```

Then, inside the file `SentimentAnalysis.ipynb`


