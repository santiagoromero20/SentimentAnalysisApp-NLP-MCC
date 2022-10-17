# Model Training Report

Abbreviations:
- B   = Balanced
- BGS = Balanced & GridSearch
- OE  = OwnEmbedding
- OEB = OwnEmbedding & Balanced
- OEBGS = OwnEmbedding & Balanced & GridSearch
- D  = Polynomial Feature Degree
- OvR = One vs Rest 


The number after a GS indicates the number of the experiment in tunning the Hyperparameters.

|   Model                        | Error_cv           | Error_tr              | Fit_score_time (sec)    |
|------------------------------|--------------------|-----------------------|--------------------|
| LogisticRegression           | 0.712 | 0.524    | 7.733  |
| LogisticRegressionB   | 0.719 | 0.657      | 3.223 |
| LogisticRegressionBGS | 0.713| 0.649    | 2.994 |
| **LogisticRegressionBGS2** | **0.684** | **0.617**    | **2.720** |
| LogisticRegressionBGS3 | 0.684 | 0.617    | 2.760 |
| DecisionTreeClassifier            | 12.79| 0.003 | 10.64|
| DecisionTreeClassifierB    | 12.95 | 0.004 | 8.857  |
| DecisionTreeClassifierBGS  | 1.049| 1.043    | 1.810 |
| DecisionTreeClassifierBGS1 | 1.031 | 1.020    | 2.013 |
| LogisticRegressionOE | 0.789 | 0.789    | 3.666 |
| LogisticRegressionOEB | 0.840 | 0.839    | 4.195 |
| LogisticRegressionOEB(D=2) | 1.031 | 1.020    | 1825 |
| DecisionTreeClassifierOEBGS | 0.874 | 0.862    | 4.017 |

**Confussion Matrix scores for our Best Model**

              precision    recall  f1-score   support

    Negative       0.73      0.72      0.73      7415
     Neutral       0.44      0.49      0.46      4107
    Positive       0.85      0.81      0.83      8929

    accuracy                            0.71     20451
    macro avg       0.67      0.67      0.67     20451
    weighted avg    0.72      0.71      0.72     20451

**Evaluation Metrics**

- "Negative" ROC AUC OvR: 0.8849
- "Neutral" ROC AUC OvR: 0.7605
- "Positive" ROC AUC OvR: 0.9220
- Average ROC AUC OvR: 0.8558

**Best Pipeline Details**


- CountVectorizer(max_features=2000, ngram_range=(1, 2))

- TfidfTransformer()

- LogisticRegression(C=1.5, class_weight='balanced', max_iter=100000,
                   multi_class='ovr', solver='liblinear')


