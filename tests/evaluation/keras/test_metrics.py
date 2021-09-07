import re
import pytest
import numpy as np
import studio.evaluation.keras.metrics as metrics

from math import log
from collections import OrderedDict


def test_metrics_top_k_multi_class(metrics_top_k_multi_class):
    concepts, y_true, y_probs = metrics_top_k_multi_class

    # 2 Correct, 2 Mistakes for top_k=1
    actual = metrics.metrics_top_k(y_probs, y_true, concepts, top_k=1)
    expected = {
        'individual': [{
            'id': 'class0',
            'label': 'class0',
            'metrics': OrderedDict([
                ('sensitivity', 1.0),
                ('precision', 0.5),
                ('f1_score', 0.6666667),
                ('TP', 1),
                ('FP', 1),
                ('FN', 0),
                ('n_samples', 1),
                ('% samples', 25.0)])},
            {
                'id': 'class1',
                'label': 'class1',
                'metrics': OrderedDict([
                    ('sensitivity', 0.0),
                    ('precision', np.nan),
                    ('f1_score', np.nan),
                    ('TP', 0),
                    ('FP', 0),
                    ('FN', 1),
                    ('n_samples', 1),
                    ('% samples', 25.0)])},
            {
                'id': 'class3',
                'label': 'class3',
                'metrics': OrderedDict([
                    ('sensitivity', 0.5),
                    ('precision', 0.5),
                    ('f1_score', 0.5),
                    ('TP', 1),
                    ('FP', 1),
                    ('FN', 1),
                    ('n_samples', 2),
                    ('% samples', 50.0)])}],
        'average': OrderedDict([
            ('accuracy', 0.5),
            ('weighted_precision', 0.375),
            ('sensitivity', 0.5),
            ('precision', 0.5),
            ('weighted_f1_score', 0.4166667),
            ('number_of_samples', 4),
            ('number_of_classes', 3),
            ('confusion_matrix', np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1]]))]
        )}

    np.testing.assert_equal(actual, expected)

    # 2 Correct, 2 Mistakes for top_k=2
    actual_accuracy = metrics.metrics_top_k(y_probs, y_true, concepts, top_k=2)['average']['accuracy']
    expected_accuracy = [0.5, 0.75]
    np.testing.assert_equal(actual_accuracy, expected_accuracy)

    actual_sensitivity = metrics.metrics_top_k(y_probs, y_true, concepts, top_k=2)
    actual_sensitivity_class_0 = actual_sensitivity['individual'][0]['metrics']['sensitivity']
    expected_sensitivity_class_0 = [1.0, 1.0]
    np.testing.assert_equal(actual_sensitivity_class_0, expected_sensitivity_class_0)

    actual_sensitivity_class_1 = actual_sensitivity['individual'][1]['metrics']['sensitivity']
    expected_sensitivity_class_1 = [0.0, 1.0]
    np.testing.assert_equal(actual_sensitivity_class_1, expected_sensitivity_class_1)

    actual_sensitivity_class_2 = actual_sensitivity['individual'][2]['metrics']['sensitivity']
    expected_sensitivity_class_2 = [0.5, 0.5]
    np.testing.assert_equal(actual_sensitivity_class_2, expected_sensitivity_class_2)

    # 2 Correct, 2 Mistakes for top_k=3
    actual = metrics.metrics_top_k(y_probs, y_true, concepts, top_k=3)['average']['accuracy']
    expected = [0.5, 0.75, 1.0]
    np.testing.assert_equal(actual, expected)

    actual_sensitivity = metrics.metrics_top_k(y_probs, y_true, concepts, top_k=3)
    actual_sensitivity_class_0 = actual_sensitivity['individual'][0]['metrics']['sensitivity']
    expected_sensitivity_class_0 = [1.0, 1.0, 1.0]
    np.testing.assert_equal(actual_sensitivity_class_0, expected_sensitivity_class_0)

    actual_sensitivity_class_1 = actual_sensitivity['individual'][1]['metrics']['sensitivity']
    expected_sensitivity_class_1 = [0.0, 1.0, 1.0]
    np.testing.assert_equal(actual_sensitivity_class_1, expected_sensitivity_class_1)

    actual_sensitivity_class_2 = actual_sensitivity['individual'][2]['metrics']['sensitivity']
    expected_sensitivity_class_2 = [0.5, 0.5, 1.0]
    np.testing.assert_equal(actual_sensitivity_class_2, expected_sensitivity_class_2)

    # Assert error when top_k <= 0 or > len(concepts)
    with pytest.raises(ValueError,
                       match=re.escape('`top_k` value should be between 1 and the total number of concepts (3)')):
        metrics.metrics_top_k(y_probs, y_true, concepts, top_k=0)

    with pytest.raises(ValueError,
                       match=re.escape('`top_k` value should be between 1 and the total number of concepts (3)')):
        metrics.metrics_top_k(y_probs, y_true, concepts, top_k=10)

    with pytest.raises(ValueError,
                       match=re.escape('`top_k` value should be between 1 and the total number of concepts (3)')):
        metrics.metrics_top_k(y_probs, y_true, concepts, top_k=-1)

    # Assert error when number of samples do not coincide
    y_true = np.asarray([0, 1, 2, 2, 1])
    with pytest.raises(ValueError,
                       match=re.escape('The number predicted samples (4) is different from the ground '
                                       'truth samples (5)')):
        metrics.metrics_top_k(y_probs, y_true, concepts, top_k=2)


def test_metrics_top_k_classes_not_present(metrics_top_k_multi_class_classes_not_present):
    concepts, y_true, y_probs = metrics_top_k_multi_class_classes_not_present

    # 2 Correct, 2 Mistakes for top_k=1
    actual = metrics.metrics_top_k(y_probs, y_true, concepts, top_k=1)

    assert actual['average']['sensitivity'] == 0.75
    assert actual['average']['precision'] == 0.75


def test_metrics_top_k_auroc():
    actual = metrics.metrics_top_k(
        y_probs=np.asarray([[0.5, 0.5], [0.5, 0.5], [0.4, 0.6]]),
        y_true=np.asarray([0, 1, 1]), concepts=[{'id': 'class0', 'label': 'class0'},
                                                {'id': 'class1', 'label': 'class1'}]
    )
    actual_auroc = actual['individual'][1]['metrics']['AUROC']
    assert actual_auroc == 0.75, actual_auroc


def test_uncertainty_distribution():
    y_probs = np.array([[0.3, 0.7], [0.67, 0.33]])
    entropy = metrics.uncertainty_distribution(y_probs)
    expected_entropy = np.array([0.88, 0.91])
    np.testing.assert_array_equal(np.round(entropy, decimals=2), expected_entropy)


def test_compute_confidence_prediction_distribution():
    y_probs = np.array([[0.3, 0.7], [0.67, 0.33]])
    confidence_prediction = metrics.compute_confidence_prediction_distribution(y_probs)
    expected_confidence = np.array([0.68, 0.32])
    np.testing.assert_array_equal(np.round(confidence_prediction, decimals=2), expected_confidence)


def test_get_correct_errors_indices():
    y_probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([1, 1, 0])

    k = [1]
    correct, errors = metrics.get_correct_errors_indices(y_probs, labels, k)
    np.testing.assert_array_equal(correct, [np.array([0, 2])])
    np.testing.assert_array_equal(errors, [np.array([1])])

    # Resilient to k being int
    k = 1
    correct, errors = metrics.get_correct_errors_indices(y_probs, labels, k)
    np.testing.assert_array_equal(correct, [np.array([0, 2])])
    np.testing.assert_array_equal(errors, [np.array([1])])

    # multiple k
    k = [1, 2]
    correct, errors = metrics.get_correct_errors_indices(y_probs, labels, k)

    np.testing.assert_array_equal(correct[0], np.array([0, 2]))
    np.testing.assert_array_equal(errors[0], np.array([1]))
    np.testing.assert_array_equal(correct[1], np.array([0, 1, 2]))
    np.testing.assert_array_equal(errors[1], np.array([]))


def test_get_top1_entropy_stats():
    y_probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([1, 1, 0])

    entropy = np.arange(0, log(y_probs.shape[1] + 0.01, 2), 0.1)
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_entropy_stats(y_probs, labels, entropy)

    np.testing.assert_array_equal(n_correct, np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]))
    np.testing.assert_array_equal(n_errors, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors

    # One value
    entropy = [0.5]
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_entropy_stats(y_probs, labels, entropy)

    np.testing.assert_array_equal(n_correct, np.array([1]))
    np.testing.assert_array_equal(n_errors, np.array([0]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors


def test_get_top1_probability_stats():
    y_probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([1, 1, 0])
    threshold = np.arange(0, 1.01, 0.1)

    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_probability_stats(y_probs, labels, threshold)

    np.testing.assert_array_equal(n_correct, np.array([2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0]))
    np.testing.assert_array_equal(n_errors, np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors

    # One value
    threshold = [0.5]
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_probability_stats(y_probs, labels, threshold)

    np.testing.assert_array_equal(n_correct, np.array([2]))
    np.testing.assert_array_equal(n_errors, np.array([1]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors


def test_confidence_interval_binomial_range():
    values = [0.01, 0.5, 0.99]
    n_samples = [100, 100, 100]

    lower, upper = metrics.confidence_interval_binomial_range(values[0], n_samples[0], 0.95)
    assert lower == 0.0
    assert round(upper, 4) == 0.0295

    lower, upper = metrics.confidence_interval_binomial_range(values[1], n_samples[1], 0.95)
    assert round(lower, 4) == 0.402
    assert round(upper, 4) == 0.598

    lower, upper = metrics.confidence_interval_binomial_range(values[2], n_samples[2], 0.95)
    assert round(lower, 4) == 0.9705
    assert round(upper, 4) == 1.0

    # Assert error when number of samples do not coincide
    with pytest.raises(ValueError, match='Confidence value not valid. '
                                         'Confidence values accepted are 0.9, 0.95, 0.98, 0.99 or 90, 95, 98, 99'):
        lower, upper = metrics.confidence_interval_binomial_range(values[2], n_samples[2], 0.123)


def test_compute_confidence_interval_binomial():
    values_a = np.array([0.01, 0.5, 0.99, 0.02, 0.55])
    values_b = np.array([0.05, 0.05, 0.55])
    probability_interval = np.array([0.01, 0.5, 0.99])
    mean, lower, upper = metrics.compute_confidence_interval_binomial(values_a, values_b,
                                                                      probability_interval=probability_interval)
    lower = [round(low, 4) for low in lower]
    upper = [round(up, 4) for up in upper]

    assert upper == [0.9605, 1.0, 1.0]
    assert lower == [0.2895, 0.3256, 1.0]
    assert mean == [0.625, 0.75, 1.0]


def test_average_precision_differential():
    y_true = np.array([[1, 1, 0], [1, 0, 0]])
    y_probs = np.array([[0.5, 0.3, 0.2], [0.7, 0.1, 0.2]])
    expected = 1.0
    actual = metrics.average_precision_differential(y_true, y_probs)
    assert actual == expected

    y_true = np.array([[0, 0, 1], [0, 1, 0]])
    expected = 0.33
    actual = metrics.average_precision_differential(y_true, y_probs)
    actual = round(actual, 2)
    assert actual == expected


def test_compute_dcg():
    # Example of https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    ranking = [3, 2, 3, 0, 1, 2]
    dcg = metrics.compute_dcg(ranking)
    expected_dcg = 6.861
    np.testing.assert_array_almost_equal(expected_dcg, dcg, 4)

    ranking_ideal = [3, 3, 2, 2, 1, 0]
    dcg_ideal = metrics.compute_dcg(ranking_ideal)
    expected_dcg_ideal = 7.141
    np.testing.assert_array_almost_equal(expected_dcg_ideal, dcg_ideal, 4)


def test_compute_ndcg():
    # Example of https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    hypothesis = [3, 2, 3, 0, 1, 2]
    reference = [3, 3, 2, 2, 1, 0]
    ndcg = metrics.compute_ndcg(hypothesis, reference)
    expected_ndcg = 0.9608
    np.testing.assert_array_almost_equal(expected_ndcg, ndcg, 4)


def test_differential_quality_scores():
    predicted_probabilities = np.array([[0.8, 0.1, 0.01, 0.09], [0.0, 0.0, 1.0, 0.0]])
    ground_truth_probabilities = np.array([[0.7, 0.3, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0]])

    # Complete match on the first sample, complete unmatch in the second sample
    differential_quality_scores = metrics.differential_quality_scores(predicted_probabilities,
                                                                      ground_truth_probabilities,
                                                                      ranked=True,
                                                                      top_x=4,
                                                                      top_y=4
                                                                      )
    expected_differential_quality_scores = [1.0, 0.0]

    np.testing.assert_array_almost_equal(expected_differential_quality_scores, differential_quality_scores, 4)

    predicted_probabilities = np.array([[0.0, 0.0, 1.0, 0.0]])
    ground_truth_probabilities = np.array([[0.6, 0.0, 0.4, 0.0]])

    # NDCG of ([0.4, 0, 0, 0], [0.6, 0.4, 0, 0])
    differential_quality_scores = metrics.differential_quality_scores(predicted_probabilities,
                                                                      ground_truth_probabilities,
                                                                      ranked=True,
                                                                      top_x=4,
                                                                      top_y=4
                                                                      )
    expected_differential_quality_scores = [0.469278]
    np.testing.assert_array_almost_equal(expected_differential_quality_scores, differential_quality_scores, 4)


def test_confusion_metrics():
    y_true = np.asarray([0, 1, 0, 1])
    y_pred = np.asarray([1, 1, 1, 0])
    results = metrics.confusion_metrics(y_true=y_true, y_pred=y_pred)

    # Compare metrics by hand.
    assert results['accuracy'] == np.mean(y_true == y_pred)
    assert results['sensitivity'] == (np.sum(np.logical_and((y_pred == 1), y_true == 1)) / np.sum(y_true))
    assert results['specificity'] == (np.sum(np.logical_and((y_pred == 0), y_true == 0)) / np.sum(y_true == 0))
    tp = np.sum(np.logical_and(np.asarray(y_true) == 1, np.asarray(y_pred) == 1))
    fp = np.sum(np.logical_and(np.asarray(y_true) == 0, (np.asarray(y_pred) == 1)))
    precision = tp / (tp + fp)
    assert results['precision'] == precision, results['precision']
