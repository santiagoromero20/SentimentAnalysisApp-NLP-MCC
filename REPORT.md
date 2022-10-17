# Model Training Report

Abbreviations:
- B   = Balanced
- BGS = Balanced & GridSearch
- OE  = OwnEmbedding
- OEB = OwnEmbedding & Balanced
- OEBGS = OwnEmbedding & Balanced & GridSearch
- D  = Polynomial Feature Degree

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


