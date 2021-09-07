import re
import pytest
import numpy as np

from studio.evaluation.keras.visualizer import plot_confusion_matrix, plot_ROC_curve, plot_precision_recall_curve,\
    plot_concept_metrics, plot_threshold, plot_models_performance, sensitivity_scatter_plot, plot_confidence_interval


def test_plot_confusion_matrix():
    confusion_matrix = np.ones((6, 5))
    concepts = ['a', 'b', 'c', 'd', 'e']
    with pytest.raises(ValueError, match='Invalid confusion matrix shape, it should be N x N.'):
        plot_confusion_matrix(confusion_matrix, concepts)

    confusion_matrix = np.ones((5, 5))
    concepts = ['a', 'b', 'c', 'd', 'e', 'f']
    with pytest.raises(ValueError, match=re.escape('Number of concepts (6) and dimensions of confusion matrix '
                                                   'do not coincide (5, 5)')):
        plot_confusion_matrix(confusion_matrix, concepts)


def test_plot_ROC_curve(metrics_top_k_multi_class):
    _, y_true_multi, y_probs_multi = metrics_top_k_multi_class
    with pytest.raises(ValueError, match='y_true must contain the true binary labels.'):
        plot_ROC_curve(y_probs_multi[:, 1], y_true_multi)


def test_plot_precision_recall_curve(metrics_top_k_multi_class):
    _, y_true_multi, y_probs_multi = metrics_top_k_multi_class
    with pytest.raises(ValueError, match='y_true must contain the true binary labels.'):
        plot_precision_recall_curve(y_probs_multi[:, 1], y_true_multi)


def test_plot_concept_metrics():
    metrics = [[0, 0, 0], [0, 0, 0]]
    concepts = ['a', 'b', 'c', 'd', 'e']
    with pytest.raises(ValueError, match=re.escape('Dimensions of concepts (5) and metrics array (2) do not match')):
        plot_concept_metrics(concepts, metrics, '', '')


def test_plot_threshold():
    th = [0, 0.1, 0.2]
    c = [12, 7, 9, 5]
    e = [2, 5, 10, 6, 6]
    with pytest.raises(ValueError, match=re.escape('The length of the arrays introduced do not coincide (3), (4), (5)')):
        plot_threshold(th, c, e)


def test_plot_models_performance(test_results_csv_paths):
    with pytest.raises(ValueError, match='Unsupported type: class_idx, metric'):
        plot_models_performance(eval_dir=test_results_csv_paths, individual=True, class_idx=0, metric=None, save_name='plot.png')


def test_plot_confidence_interval():
    values_x = [0, 1]
    values_y = [0, 1]
    lower = [0, 1]
    upper = [0]
    with pytest.raises(ValueError, match='Arrays "values_x", "values_y", "lower_bound" and "upper_bound" '
                                         'should have the same dimension'):
        plot_confidence_interval(values_x, values_y, lower, upper)


def test_sensitivity_scatter_plot():
    values_x = [0, 1]
    values_y = [0]
    with pytest.raises(ValueError, match='Both arrays "values_x" and "values_y" should have the same dimension'):
        sensitivity_scatter_plot(values_x, values_y, [], '', '', '')
