# Utility functions meant to aid our convolutional neural network.

from scipy import stats

import pandas            as pb
import numpy             as np
import matplotlib.pyplot as plt
import tensorflow        as tf


%matplotlib inline
plt.style.use('ggplot')

def read_data(file_path):
    '''
    TODO: Documentation
    '''
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data

def feature_normalize(dataset):
    '''
    TODO: Documentation
    '''
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)

    return (dataset - mu)/sigma

def plot_axis(ax, x, y, title):
    '''
    TODO: Documentation
    '''
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

def plot_activity(activity, data):
    '''
    TODO: Documentation.
    '''
    fig, (ax0, ax1, ax2) = plt.subplots(ntows = 3, figsize = (15, 10), sharex=True)

    # Plot the axis that is in use.
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')

    # Figure processing.
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

    
