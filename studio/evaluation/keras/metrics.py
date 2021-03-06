from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.stats

from math import log, sqrt
from studio.evaluation.keras import utils
from collections import OrderedDict
from keras.utils.np_utils import to_categorical
from studio.evaluation.keras.metrics_utils import create_ndcg_inputs
from sklearn.metrics import confusion_matrix, roc_curve, average_precision_score


def metrics_top_k(y_probs, y_true, concepts, top_k=1, round_decimals=7):
    """Compute metrics between the predicted probabilities `y_probs` and the true labels `y_true`.

    Notes:
        The precision is only computed when k=1, since it is not clear how to compute when k>1.
        np.nan values are used for undefined values (i.e., divide by zero errors).
            This occurs if `y_probs` does not predict a certain class,
            and/or if there there are no values in `y_true` for a certain class.

    Args:
        y_probs: a numpy array of the class probabilities.
        y_true: a numpy array of the true class labels (*not* encoded as 1-hot).
        concepts: a list of dictionaries ({id:x, label:y}) containing the `id` and `label` of the classes.
        top_k: a number specifying the top-k results to compute. E.g. 2 will compute top-1 and top-2
        round_decimals: Integer indicating the number of decimals to rounded.

    Returns:
        metrics: a dictionary of lists of dictionaries containing the results.
            The dict keys correspond to the input `labels`.
            For clarity, see the tests in `tests/test_metrics/test_metrics_top_k()`

    """
    utils.check_input_samples(y_probs, y_true)
    utils.check_top_k_concepts(concepts, top_k)

    accuracy_k = []
    class_precision = []
    weighted_class_precision = []
    class_sensitivity = []
    average_f1_score = []

    top_k_sensitivity = []
    top_k_sensitivity_dict = []
    top_k_array = np.arange(1, top_k + 1, 1)
    one_hot_y_true = to_categorical(y_true, num_classes=len(concepts))

    # Sort predictions from higher to smaller and get class indices
    top_preds = y_probs.argsort(axis=1)[:, ::-1]
    total_samples = y_true.shape[0]

    metrics = {'average': {}, 'individual': []}

    # top-K Accuracy
    for k in top_k_array:
        # Select k top predictions
        top_k_preds = top_preds[:, 0:k]
        # Compute the top-k matches
        matches_k = (top_k_preds == y_true[:, np.newaxis])
        in_top_k = np.sum(matches_k, axis=1) > 0
        accuracy_k.append(np.sum(in_top_k) / float(len(in_top_k)))

        for idx, concept in enumerate(concepts):
            total_samples_concept = np.sum(y_true == idx)
            tp_top_k = np.sum(in_top_k[(y_true == idx)])
            sensitivity = round(utils.safe_divide(float(tp_top_k), total_samples_concept), round_decimals)
            top_k_sensitivity_dict.append({'concept': concept['id'], 'k': k, 'sensitivity': sensitivity})

    # Top-1 metrics
    one_hot_top_1_preds = to_categorical(top_preds[:, 0:1], num_classes=len(concepts))

    for idx, concept in enumerate(concepts):
        total_samples_concept = np.sum(y_true == idx)
        percentage_samples_concept = round(total_samples_concept / len(y_true) * 100, 2)

        tn, fp, fn, tp = confusion_matrix(one_hot_y_true[:, idx], one_hot_top_1_preds[:, idx], labels=[0, 1]).astype(
            np.float32).ravel()

        concept_sensitivity = []
        for dict_item in top_k_sensitivity_dict:
            if dict_item['concept'] == concept['id']:
                concept_sensitivity.append(dict_item['sensitivity'])
        top_k_sensitivity.append(concept_sensitivity)

        sensitivity = top_k_sensitivity[idx]

        if not np.isnan(sensitivity[0]):
            class_sensitivity.append(sensitivity[0])

        precision = round(utils.safe_divide(tp, tp + fp), round_decimals)

        if not np.isnan(precision):
            weighted_class_precision.append(precision * total_samples_concept)
            class_precision.append(precision)

        f1_score = round(2 * utils.safe_divide(precision * sensitivity[0], precision + sensitivity[0]), round_decimals)
        if not np.isnan(f1_score):
            average_f1_score.append(f1_score * total_samples_concept)

        sensitivity = utils.round_list(sensitivity, round_decimals)
        metrics_dict = OrderedDict([('sensitivity', sensitivity if len(sensitivity) > 1 else sensitivity[0]),
                                    ('precision', precision), ('f1_score', f1_score)])

        # Binary classification metrics
        if len(concepts) == 2:
            specificity = round(utils.safe_divide(tn, tn + fp), round_decimals)
            fdr = round(utils.safe_divide(fp, tp + fp), round_decimals)
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, idx], pos_label=idx)
            auroc = round(np.trapz(tpr, fpr), round_decimals)

            metrics_dict.update([('specificity', specificity), ('FDR', fdr), ('AUROC', auroc)])

        metrics_dict.update(
            [('TP', int(tp)), ('FP', int(fp)), ('FN', int(fn)), ('n_samples', total_samples_concept),
             ('% samples', percentage_samples_concept)])

        metrics['individual'].append({'id': concept['id'], 'label': concept['label'], 'metrics': metrics_dict})

    accuracy = utils.round_list(accuracy_k, round_decimals)

    metrics['average'] = OrderedDict([('accuracy', accuracy if len(accuracy) > 1 else accuracy[0]),
                                      ('weighted_precision', round(utils.safe_divide(sum(weighted_class_precision),
                                                                                     total_samples), round_decimals)),
                                      ('sensitivity', round(utils.safe_divide(sum(class_sensitivity),
                                                                              len(class_sensitivity)), round_decimals)),
                                      ('precision', round(utils.safe_divide(sum(class_precision), len(class_precision)),
                                                          round_decimals)),
                                      ('weighted_f1_score', round(utils.safe_divide(sum(average_f1_score),
                                                                                    total_samples), round_decimals)),
                                      ('number_of_samples', total_samples),
                                      ('number_of_classes', len(concepts)),
                                      ('confusion_matrix', confusion_matrix(y_true, top_preds[:, 0],
                                                                            labels=np.arange(0, len(concepts))))
                                      ])

    return metrics


def compute_confidence_prediction_distribution(y_probs, verbose=1):
    if y_probs.ndim == 3:
        if y_probs.shape[0] <= 1:
            y_probs = y_probs[0]
            prob_mean = np.mean(np.sort(y_probs)[:, ::-1], axis=0)
    elif y_probs.ndim == 2:
        prob_mean = np.mean(np.sort(y_probs)[:, ::-1], axis=0)
    else:
        raise ValueError('Incorrect shape for `y_probs` array, we accept [n_samples, n_classes] or '
                         '[n_models, n_samples, n_classes]')
    if verbose == 1:
        for ind, prob in enumerate(prob_mean):
            print('Confidence mean at giving top %i prediction is %f' % (ind + 1, prob))

    return prob_mean


def uncertainty_distribution(y_probs, combination_mode=None, verbose=1):
    '''
    Args:
        y_probs: y_probs after model forwarding
        verbose: Show text
        combination_mode: Ensemble combination mode

    Returns: The entropy for each of the predictions given [n_images]
    '''

    y_probs = utils.combine_probabilities(y_probs, combination_mode)
    entropy = scipy.stats.entropy(y_probs.T, base=2.0)

    if verbose == 1:
        print('There are %i classes ' % y_probs.shape[1])
        print('Max uncertainty value is %.3f' % log(y_probs.shape[1], 2))
        print('The mean entropy or uncertainty per image is %.3f' % np.mean(entropy))
    return entropy


def get_correct_errors_indices(y_probs, y_true, k, split_k=False, verbose=1):
    '''
    Args:
        y_probs: y_probs of the model / ensemble [n_images, n_class] / [n_models, n_images, n_class]
        y_true:  Ground truth [n_images]
        k: Top k y_probs to compute the errors / correct
        split_k: If true, not consider top-n being n<k to compute the top-k correct/error predictions
        combination_mode: For ensembling y_probs

    Returns: Returns A list containing for each of the values of k provided, the indices of the images
            with errors / correct and the y_probs in format [n_images,n_class].
    '''
    if k is None:
        raise ValueError('k is required')
    k_list = [k] if not isinstance(k, (list, tuple, np.ndarray)) else k

    errors = []
    correct = []

    for k_val in k_list:

        probabilities_class = y_probs.argsort(axis=1)[:, -k_val:]

        # Get True/False values for predictions that match.
        preds_match = probabilities_class == y_true[:, np.newaxis]

        # Find the positions of the errors / correct
        if split_k:
            # the top-k prediction is the first in each row top-k top-(k-1) ... top-1
            errors.append(np.where(preds_match[:, 0] < 1)[0])
            correct.append(np.where(preds_match[:, 0] > 0)[0])
        else:
            errors.append(np.where(np.sum(preds_match, axis=1) < 1)[0])
            correct.append(np.where(np.sum(preds_match, axis=1) > 0)[0])
    if verbose:
        print('Returning correct predictions and errors for the top k: ', k_list)
    return correct, errors


def get_top1_probability_stats(y_probs, y_true, threshold, verbose=0):
    '''
    Args:
        y_probs: y_probs of the model / ensemble [n_images, n_class] / [n_models, n_images, n_class]
        y_true: Ground truth [n_images]
        threshold: Value or set of values, can be list, numpy array or tuple
        combination_mode:  For ensembling y_probs
        verbose: Show text

    Returns: A list that for each threshold introduced, has the indices of the images in which we have had an error
    classifying them (top-1) k = 1

    '''
    # Get top-1 errors and correct predictions
    correct, errors = get_correct_errors_indices(y_probs, y_true, k=1)
    correct = correct[0]
    errors = errors[0]
    # Get the y_probs associated
    error_probabilities = y_probs[errors]
    correct_probabilities = y_probs[correct]
    # Get the top probability
    top_error_probability = np.sort(error_probabilities, axis=1)[:, ::-1][:, 0]
    top_correct_probability = np.sort(correct_probabilities, axis=1)[:, ::-1][:, 0]

    # If we input a threshold / set of thresholds extract the indices of the images that we will misclassify
    if threshold is None:
        raise ValueError('th is required')
    threshold_list = [threshold] if not isinstance(threshold, (list, tuple, np.ndarray)) else threshold

    # Compute indices of the errors over certain th
    errors_list = []
    # Compute indices of the correct over certain th
    correct_list = []

    # number of correct / error predictions
    n_correct = []
    n_errors = []

    for i, thres in enumerate(threshold_list):

        errors_over_th = [errors[i] for i, prob in enumerate(top_error_probability) if prob > thres]
        correct_over_th = [correct[i] for i, prob in enumerate(top_correct_probability) if prob > thres]

        errors_over_th = np.array(errors_over_th)
        correct_over_th = np.array(correct_over_th)
        n_correct.append(correct_over_th.shape[0])
        n_errors.append(errors_over_th.shape[0])
        errors_list.append(errors_over_th)
        correct_list.append(correct_over_th)

        if verbose == 1:
            print('-- For a threshold value of  %.2f --\n' % thres)
            print('The total number of errors was %i\nThere were %i errors that will be classified as correct\n'
                  % (errors.shape[0], errors_over_th.shape[0]))

            print('%.3f%% of the errors will be removed\n'
                  % (((errors.shape[0] - errors_over_th.shape[0]) / errors.shape[0]) * 100))

            print('The total number of correct predictions was %i\nThere are %i that will be predicted as correct\n'
                  % (correct.shape[0], correct_over_th.shape[0]))

            print('%.3f%% of the correct predictions will be predicted\n'
                  % ((correct_over_th.shape[0] / correct.shape[0]) * 100))

    n_correct = np.array(n_correct)
    n_errors = np.array(n_errors)

    return correct_list, errors_list, n_correct, n_errors


def get_top1_entropy_stats(y_probs, y_true, entropy, verbose=0):
    '''
    Args:
        y_probs: y_probs of the model / ensemble [n_images, n_class] / [n_models, n_images, n_class]
        y_true: Ground truth [n_images]
        entropy: Value or set of values, can be list or numpy array with max entropy values
        combination_mode: For ensembling y_probs
        verbose: Show text

    Returns: A list that for each entropy value introduced, has the indices of the images in which we have had an error
    classifying them (top-1) k = 1

    '''
    # Get top-1 errors and correct predictions
    correct, errors = get_correct_errors_indices(y_probs, y_true, k=1)
    correct = correct[0]
    errors = errors[0]
    # Get the entropy associated
    probabilities_entropy = uncertainty_distribution(y_probs)
    error_entropy = probabilities_entropy[errors]
    correct_entropy = probabilities_entropy[correct]

    # If we input a threshold / set of thresholds extract the indices of the images that we will misclassify
    if entropy is None:
        raise ValueError('ent is required')
    entropy_list = [entropy] if not isinstance(entropy, (list, tuple, np.ndarray)) else entropy

    # Compute indices of the errors over certain entropy
    errors_list = list()
    # Compute indices of the correct over certain entropy
    correct_list = list()

    # number of correct / error predictions
    n_correct = []
    n_errors = []

    for i, ent in enumerate(entropy_list):
        errors_below_e = [errors[i] for i, entropy_value in enumerate(error_entropy) if entropy_value < ent]

        correct_below_e = [correct[i] for i, entropy_value in enumerate(correct_entropy) if entropy_value < ent]

        errors_below_e = np.array(errors_below_e)
        correct_below_e = np.array(correct_below_e)
        n_correct.append(correct_below_e.shape[0])
        n_errors.append(errors_below_e.shape[0])

        errors_list.append(errors_below_e)
        correct_list.append(correct_below_e)

        if verbose == 1:
            print('-- For a entropy value of  %.2f --\n' % ent)
            print('The total number of errors was %i\nThere were %i errors that will be classified as correct\n'
                  % (errors.shape[0], errors_below_e.shape[0]))

            print('%.3f%% of the errors will be removed\n'
                  % (((errors.shape[0] - errors_below_e.shape[0]) / errors.shape[0]) * 100))

            print('The total number of correct predictions was %i\nThere are %i that will be predicted as correct\n'
                  % (correct.shape[0], correct_below_e.shape[0]))

            print('%.3f%% of the correct predictions will be predicted\n'
                  % ((correct_below_e.shape[0] / correct.shape[0]) * 100))

    n_correct = np.array(n_correct)
    n_errors = np.array(n_errors)

    return correct_list, errors_list, n_correct, n_errors


def confidence_interval_binomial_range(value, samples, confidence=0.95):
    '''
    Computes lower and upper bounds for a determined confidence interval given the mean value following a
    binomial distribution using the gaussian approximation.
    Args:
        value: Mean value
        samples: Number of observations
        confidence: Percentage of confidence. Values accepted are 0.9, 0.95, 0.98, 0.99 or 90, 95, 98, 99

    Returns: Lower and upper bounds.

    '''
    confidence_cts = {0.90: 1.64, 0.95: 1.96, 0.98: 2.33, 0.99: 2.58, 90: 1.64, 95: 1.96, 98: 2.33, 99: 2.58}
    accepted_confidence_keys = confidence_cts.keys()
    if confidence in accepted_confidence_keys:
        val = confidence_cts[confidence] * sqrt((value * (1 - value)) / samples)
        return max(0.0, value - val), min(value + val, 1.0)
    else:
        raise ValueError('Confidence value not valid.'
                         ' Confidence values accepted are 0.9, 0.95, 0.98, 0.99 or 90, 95, 98, 99')


def compute_confidence_interval_binomial(values_a, values_b, confidence=0.95,
                                         probability_interval=np.arange(0, 1.01, 0.01)):
    '''
    Computes mean, lower and upper bounds for a determined confidence value for a given success and failure values
    following a binomial distribution using the gaussian approximation.
    Args:
        values_a: Success values. Probabilities at which we had a success outcome.
        values_b: Failure values. Probabilities at which we had a failure outcome.
        confidence: Percentage of confidence. Values accepted are 0.9, 0.95, 0.98, 0.99 or 90, 95, 98, 99
        probability_interval: Probabilities to compare with.

    Returns:

    '''
    mean = []
    lower_bound = []
    upper_bound = []
    counts_a = np.zeros((len(probability_interval)))
    counts_b = np.zeros((len(probability_interval)))
    for i, val in enumerate(probability_interval):
        for val_a in values_a:
            if val_a >= val:
                counts_a[i] += 1
        for val_b in values_b:
            if val_b >= val:
                counts_b[i] += 1
        samples = counts_a[i] + counts_b[i]
        mean.append(counts_a[i] / samples)
        lower, upper = confidence_interval_binomial_range(mean[i], samples, confidence)
        lower_bound.append(lower)
        upper_bound.append(upper)
    return mean, lower_bound, upper_bound


def average_precision_differential(y_true, y_probs):
    '''
    Finds the average precision with multiple true labels
    Args:
        y_true: Numpy array of a list of lists with true labels
        y_probs: Numpy array of list of lists with the probabilities of predictions
    Returns:
        The average precision across the test set.
    '''
    precision_list = []
    for i in range(y_true.shape[0]):
        precision_list.append(average_precision_score(y_true[i], y_probs[i]))
    return np.mean(precision_list)


def differential_quality_scores(predicted_probabilities,
                                ground_truth_probabilities,
                                ranked=True,
                                top_x=5,
                                top_y=5,
                                min_probability=0.05):
    """

    Args:
        predicted_probabilities: Numpy array of ML model probabilities.
        ground_truth_probabilities: Numpy array of ground truth probabilities.
        ranked: If True will compute ranked reference, else unranked.
        top_x: First X diagnosis in the sorted predicted probabilities used to build the hypothesis array.
        top_y: First Y diagnosis in the sorted ground truth probabilities used to build the references array.
        min_probability: Minimum probability of a ground truth diagnosis to be included inside the references.

    Returns:
        The differential quality score (NDCG) of the samples
    """
    references_scores, references_indices, hypothesis = create_ndcg_inputs(predicted_probabilities,
                                                                           ground_truth_probabilities,
                                                                           ranked=ranked,
                                                                           top_x=top_x,
                                                                           top_y=top_y,
                                                                           min_probability=min_probability)
    differential_quality_scores = []
    for i in range(len(references_scores)):
        differential_quality_scores.append(compute_ndcg(hypothesis[i], references_scores[i]))
    return differential_quality_scores


def compute_ndcg(hypothesis, reference):
    """
    Normalized Discounted Cumulative Gain (nDCG)
    Normalized version of DCG: nDCG = DCG(hypothesis)/DCG(reference)
    Args:
        reference: A gold standard (perfect) ordering Ex: [5,4,3,2,1]
        hypothesis: a proposed ordering Ex: [5,2,2,3,1]
    Returns:
        ndcg_score: normalized score
    """

    return compute_dcg(hypothesis) / compute_dcg(reference)


def compute_dcg(element_list):
    """
    Discounted Cumulative Gain (DCG) -- https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    Args:
        element_list: a list of ranks Ex: [5,4,2,2,1]
    Returns:
        DCG score
    """
    score = 0.0
    for order, rank in enumerate(element_list):
        score += float(rank) / log((order + 2), 2)
    return score


def confusion_metrics(y_true, y_pred):
    """Compute metrics for a binary confusion matrix.

    Args:
        y_true: A numpy array or list of N-binary true labels.
        y_pred: A numpy array or list of N-binary predicted labels.

    Returns:
        A dictionary of common metrics derived from a binary confusion matrix.

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    accuracy = np.mean(y_true == y_pred)

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }


def binary_confusion_matrix_elements(true_labels, pred_labels, positive_label):
    # Ensure is cast to array for `==` and `!=` operations to compute element-wise.
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)

    # True Positive: we predict a label of `positive_label`, and the true label is `positive_label`.
    tp = np.sum(np.logical_and(pred_labels == positive_label, true_labels == positive_label))

    # True Negative: we predict a negative label, and the true label is negative.
    # Note for multi-label problems, it's not clear what is a "negative label".
    # Here we assume a negative label to be anything that is not the `positive_label`.
    # In multi-label problems, a high true-negative count may be misleading.
    tn = np.sum(np.logical_and(pred_labels != positive_label, true_labels != positive_label))

    # False Positive: we predict a positive label, but the true label is not the positive label.
    fp = np.sum(np.logical_and(pred_labels == positive_label, true_labels != positive_label))

    # False Negative: we predict a negative label (i.e., not a positive label), but the true label is positive
    fn = np.sum(np.logical_and(pred_labels != positive_label, true_labels == positive_label))

    return np.asarray((tn, fp, fn, tp))
