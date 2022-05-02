import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes


def plot_simple_graph(x, y, x_label, y_label, title, ax: Axes = None):
    """
    given 2d data plots a simple graph where the indices of the data are the x axis and the values of the
    data are the y axis.
    :param title:
    :param ax:
    :param data:
    :param x_label: the label of the x axis
    :param y_label: the label of the y axis
    :return:
    """
    has_ax = ax is not None
    if not has_ax:
        _, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.title.set_text(title)
    if not has_ax:
        plt.show()


def plot_graphs(data_dict, n_rows, n_columns, x_label, y_label, row_size=6, col_size=6):
    """

    :param n_rows:
    :param n_columns:
    :param x_label:
    :param y_label:
    :param row_size:
    :param col_size:
    :param data_dict: key is title and value is the data to plot
    :return:
    """
    f, axs = plt.subplots(n_rows, n_columns, figsize=(row_size * n_rows, col_size * n_columns))

    for i, ax in enumerate(axs.reshape(-1)):
        current_values = list(data_dict.items())[i]
        plot_simple_graph(current_values[1][0], current_values[1][1], x_label=x_label[i], y_label=y_label[i],
                          title=current_values[0], ax=ax)
    plt.show()


def plot_roc_curve(fpr_tpr_dict: dict):
    """
    Plots the roc curve of the given estimators rates with the baseline
    :param fpr_tpr_dict: a dictionary where the key is the name of the estimator and the value is a
    dictionary with tpr and fpr values
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title("ROC curve")
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.plot(np.linspace(0, 1), np.linspace(0, 1), label="baseline", linestyle='--')
    for estimator, rates in fpr_tpr_dict.items():
        try:
            ax.plot(rates["fpr"], rates["tpr"], label=estimator)
        except KeyError:
            raise Exception(f"The keys are expected to be fpr and tpr for {estimator} rates")
    ax.legend()
    plt.show()
