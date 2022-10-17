import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
import time

#---------------------------------------------MODEL EVALUATION---------------------------------------------#

"""
Approch to compute AUC score of multi class classification problem:

OBS: This metrics tells how much the model is capable of distinguishing between classes. 
Higher the AUC, the better the model is at predicting 0s as 0s and 1s as 1s...

.One vs All Technique : It involves splitting the multi-class dataset into multiple binary classification problems. 
A binary classifier is then trained on each binary classification problem and predictions are made using the model that is the most confident.

One vs One Technique : It is really similar to OvR, but instead of comparing each class with the rest,
we compare all possible two-class combinations of the dataset.

We will be using Ons vs All Technique as it is a little bit "friendlier" to plot (less images) and is good enough for our purposes
""" 

#AUXILIARIES

def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations
    
    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes
        
    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''
    # Calculates the confusion matrix and recover each element
    cm = metrics.confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr

def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a treshold for the predicion of the class.
    
    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.
        
    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list
    
def plot_roc_curve(tpr, fpr, scatter = True, ax = None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).
    
    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    if ax == None:
        plt.figure(figsize = (5, 5))
        ax = plt.axes()
    
    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
 

#ONE VS REST

def plot_roc_auc_score_multiclass_ovr(model, X_test, y_test):

    #Neccesaries Tranformations
    y_test = y_test["Rating"].values.tolist()

    #Starting Plotting and Calculations
    plt.figure(figsize = (12, 8))
    bins = [i/20 for i in range(20)] + [1]
    classes = model.classes_
    roc_auc_ovr = {}
    for i in range(len(classes)):

        # Gets the class
        c = classes[i]
        
        # Prepares an auxiliar dataframe to help with the plots
        y_proba = model.predict_proba(X_test)
        df_aux = pd.DataFrame()
        df_aux['class'] = [1 if y == c else 0 for y in y_test]
        df_aux['prob'] = y_proba[:, i]
        df_aux = df_aux.reset_index(drop = True)
        
        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, 3, i+1)
        sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
        ax.set_title(c)
        ax.legend([f"Class: {c}", "Rest"])
        ax.set_xlabel(f"P(x = {c})")
        

        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(2, 3, i+4)
        tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
        plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
        ax_bottom.set_title("ROC Curve OvR")

        # Calculates the ROC AUC OvR
        roc_auc_ovr[c] = metrics.roc_auc_score(df_aux['class'], df_aux['prob'])

    plt.tight_layout()
    

    # Displays the ROC AUC for each class
    avg_roc_auc = 0
    i = 0
    for k in roc_auc_ovr:
        avg_roc_auc += roc_auc_ovr[k]
        i += 1
        print(f"{k} ROC AUC OvR: {roc_auc_ovr[k]:.4f}")

    ans = avg_roc_auc/i
    print(f"Average ROC AUC OvR: {ans:.4f}")
 
def get_avg_roc_auc(model, X_test, y_test):

    #Neccesaries Tranformations
    y_test = y_test["Rating"].values.tolist()
    classes = model.classes_
    roc_auc_ovr = {}
    for i in range(len(classes)):

        # Gets the class
        c = classes[i]
        
        # Prepares an auxiliar dataframe to help with the plots
        y_proba = model.predict_proba(X_test)
        df_aux = pd.DataFrame()
        df_aux['class'] = [1 if y == c else 0 for y in y_test]
        df_aux['prob'] = y_proba[:, i]
        df_aux = df_aux.reset_index(drop = True)
        roc_auc_ovr[c] = metrics.roc_auc_score(df_aux['class'], df_aux['prob'])
    
    # Displays the ROC AUC for each class
    avg_roc_auc = 0
    i = 0
    for k in roc_auc_ovr:
        avg_roc_auc += roc_auc_ovr[k]
        i += 1

    ans = avg_roc_auc/i
    return ans

#---------------------------------------------LEARNING CURVES AND TUNNING---------------------------------------------#

#Auxiliaries functions without GS

def calculating_error(dict, key_error):
    error_array = -dict[str(key_error)]
    error = np.mean(error_array)
    return error

def cv_calculus(pipe, X_train, y_train, scoring, cv):

    #Perform the CV
    cv_results = cross_validate(pipe, X_train, y_train, scoring=scoring, cv=cv, return_train_score=True) #return_train_score=True remains harcoded

    #Clean CV Results
    error_cv = calculating_error(cv_results, "test_score")
    error_tr = calculating_error(cv_results, "train_score")
    fit_score_time = (cv_results["fit_time"].mean()) + (cv_results["score_time"].mean())

    #Print the Main results
    print(f"Training Log Loss Error {error_tr:0.2f}, CV Log Loss Error {error_cv:0.2f}")
    print('Train/Validation: {}'.format(round(error_cv/error_tr, 1)))

    return error_cv, error_tr, fit_score_time

#Auxiliary functions when GridSearch is performed

def get_errors(grid_results):
    results = pd.DataFrame.from_dict(grid_results)
    results = results.sort_values(by=["mean_test_score"], ascending=False)
    results = results.head().reset_index()
    error_cv = -results.loc[0, "mean_test_score"]
    error_tr = -results.loc[0, "mean_train_score"]
    fit_score_time = (results.loc[0,"mean_fit_time"]) + (results.loc[0,"mean_score_time"])
    print(f"Training Log Loss {error_tr:0.2f}, CV Log Loss {error_cv:0.2f}")
    print('Train/Validation: {}'.format(round(error_cv/error_tr, 1)))
    return error_cv, error_tr, fit_score_time

def grid_search(estimator, parameters, cv, scoring, X_train, y_train):
    grid = GridSearchCV(estimator, param_grid=parameters, cv=cv, scoring=str(scoring), return_train_score=True)
    inicio = time.time()
    grid.fit(X_train, y_train)
    fin = time.time()
    print("The time it takes to fit the model is",round(fin-inicio),"seconds.")
    print("Best params: "+str(grid.best_params_))
    return grid