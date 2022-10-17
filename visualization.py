
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')

import random
random.seed(0)

#---------------------------------------------MODEL EVALUATION---------------------------------------------#

def plot_learning_curve(train_sizes, train_scores, validation_scores, ylabel, xlabel, title):

    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1)

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, color="blue",label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, color="green",label = 'Validation error')
    plt.ylabel(str(ylabel), fontsize = 14)
    plt.xlabel(str(xlabel), fontsize = 14)
    plt.title(str(title), fontsize = 18, y = 1.03)
    plt.legend()


#---------------------------------------------DIMENSIONALITY REDUCTION--------------------------------------------#

def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    
    # Label randomly subsampled 25 data points
    
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


