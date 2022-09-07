import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def get_performance(predictions, y_test, labels=[1, 0]):
    accuracy = metrics.accuracy_score(y_test, predictions)
    precision = metrics.precision_score(y_test, predictions, average='micro')
    recall = metrics.recall_score(y_test, predictions, average='micro')
    f1_score = metrics.f1_score(y_test, predictions, average='micro')
    
    report = metrics.classification_report(y_test, predictions)
    
    cm = metrics.confusion_matrix(y_test, predictions)
    cm_as_dataframe = pd.DataFrame(data=cm)
    
    print('Model Performance metrics:')
    print('-'*30)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score)
    print('\nModel Classification report:')
    print('-'*30)
    print(report)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    print(cm_as_dataframe, "\n")
    
    return accuracy, precision, recall, f1_score


"""
Approch to compute AUC score of multi class classification problem:

OBS: This metrics tells how much the model is capable of distinguishing between classes. 
Higher the AUC, the better the model is at predicting 0s as 0s and 1s as 1s...

.One vs All Technique : It involves splitting the multi-class dataset into multiple binary classification problems. 
A binary classifier is then trained on each binary classification problem and predictions are made using the model that is the most confident.

One vs One Technique : It is really similar to OvR, but instead of comparing each class with the rest,
we compare all possible two-class combinations of the dataset.
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
    #print("como")
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
    #print("hola")
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
    #print("estas")
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
        print("Class:", c)
        
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
    return ans


"""
#ONE VS ONE

def plot_roc_auc_multiclass_ovo(model, X_test, y_test):

    #Get Classes Combinations
    classes = model.classes_
    classes_combinations = []
    class_list = list(classes)
    for i in range(len(class_list)):
        for j in range(i+1, len(class_list)):
            classes_combinations.append([class_list[i], class_list[j]])
            classes_combinations.append([class_list[j], class_list[i]])
    
    # Plots the Probability Distributions and the ROC Curves One vs One
    plt.figure(figsize = (20, 7))
    bins = [i/20 for i in range(20)] + [1]
    roc_auc_ovo = {}
    for i in range(len(classes_combinations)):
        # Gets the class
        comb = classes_combinations[i]
        c1 = comb[0]
        c2 = comb[1]
        c1_index = class_list.index(c1)
        title = c1 + " vs " +c2
        
        # Prepares an auxiliar dataframe to help with the plots
        y_proba = model.predict_proba(X_test)
        df_aux = X_test.copy()
        df_aux['class'] = y_test
        df_aux['prob'] = y_proba[:, c1_index]
        
        # Slices only the subset with both classes
        df_aux = df_aux[(df_aux['class'] == c1) | (df_aux['class'] == c2)]
        df_aux['class'] = [1 if y == c1 else 0 for y in df_aux['class']]
        df_aux = df_aux.reset_index(drop = True)
        
        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, 6, i+1)
        sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
        ax.set_title(title)
        ax.legend([f"Class 1: {c1}", f"Class 0: {c2}"])
        ax.set_xlabel(f"P(x = {c1})")
        
        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(2, 6, i+7)
        tpr, fpr = metrics.get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
        metrics.plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
        ax_bottom.set_title("ROC Curve OvO")
        
        # Calculates the ROC AUC OvO
        roc_auc_ovo[title] = metrics.roc_auc_score(df_aux['class'], df_aux['prob'])
    plt.tight_layout()

    #Let´s also get the Roc-Auc value of the different classes
    avg_roc_auc = 0
    i = 0
    for k in roc_auc_ovo:
        avg_roc_auc += roc_auc_ovo[k]
        i += 1
        print(f"{k} ROC AUC OvO: {roc_auc_ovo[k]:.4f}")
    print(f"average ROC AUC OvO: {avg_roc_auc/i:.4f}")
"""



