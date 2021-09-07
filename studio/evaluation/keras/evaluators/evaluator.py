import numpy as np
import pandas as pd
import studio.evaluation.keras.utils as utils
import studio.evaluation.keras.metrics as metrics
import studio.evaluation.keras.visualizer as visualizer

from math import log
from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def get_metrics(self, probabilities, labels, top_k=1, concepts=None, filter_indices=None,
                    confusion_matrix=False, save_confusion_matrix_path=None, show_confusion_matrix_text=True):
        """Compute and return metrics for the predicted `probabilities` and the `labels`.

        Args:
            probabilities:  Predicted probabilities with a shape of (n_samples, n_classes)
            labels: Ground truth labels of length (n_samples)
            top_k: The top-k predictions to consider. E.g. top_k = 5 is top-5 preds
            concepts: List containing the concept_labels
            filter_indices: If given, compute only the predictions corresponding to the specified indices
            confusion_matrix: If True, show the confusion matrix
            save_confusion_matrix_path: If path specified, save confusion matrix there
            show_confusion_matrix_text: If False, will hide the text in the confusion matrix

        Returns: A dictionary with metrics for each concept

        """

        concept_labels = utils.get_concept_items(concepts, key='label')

        if filter_indices is not None:
            probabilities = probabilities[filter_indices]
            labels = labels[filter_indices]

        # Print sensitivity and precision for different values of K.
        self.results = metrics.metrics_top_k(probabilities, y_true=labels, concepts=concepts, top_k=top_k)

        # Show metrics visualization as a confusion matrix
        if confusion_matrix:
            self.plot_confusion_matrix(confusion_matrix=self.results['average']['confusion_matrix'],
                                       concept_labels=concept_labels, save_path=save_confusion_matrix_path,
                                       show_text=show_confusion_matrix_text)

        return self.results

    def get_image_paths_by_prediction(self, probabilities, labels, concept_labels=None, image_paths=None):
        """
        Return the list of images given its predictions.
        Args:
            probabilities: Probabilities given by the model [n_samples,n_classes]
            labels: Ground truth labels (categorical)
            concept_labels: List with class names (by default last evaluation)
            image_paths: List with image_paths (by default last evaluation)

        Returns: A dictionary containing a list of images per confusion matrix square (relation ClassA_ClassB), and the
        predicted probabilities

        """

        if image_paths is None:
            image_paths = self.image_paths

        if labels is None:
            labels = self.labels

        if probabilities.shape[0] != len(image_paths):
            raise ValueError('Length of probabilities (%i) do not coincide with the number of image paths (%i)' %
                             (probabilities.shape[0], len(image_paths)))

        if concept_labels is None:
            concept_labels = utils.get_concept_items(self.concepts, key='label')

        predictions = np.argmax(probabilities, axis=1)
        dict_image_paths_concept = {}

        for name_1 in concept_labels:
            for name_2 in concept_labels:
                if name_1 == name_2:
                    dict_image_paths_concept.update({name_1 + '_' + name_2: {'image_paths': [], 'probs': [],
                                                                             'diagonal': True}})
                else:
                    dict_image_paths_concept.update({name_1 + '_' + name_2: {'image_paths': [], 'probs': [],
                                                                             'diagonal': False}})

        for i, pred in enumerate(predictions):
            predicted_label = concept_labels[pred]
            correct_label = concept_labels[labels[i]]
            list_image_paths = dict_image_paths_concept[str(correct_label + '_' + predicted_label)]['image_paths']
            list_image_paths.append(image_paths[i])
            list_probs = dict_image_paths_concept[str(correct_label + '_' + predicted_label)]['probs']
            list_probs.append(probabilities[i])
            diagonal = dict_image_paths_concept[str(correct_label + '_' + predicted_label)]['diagonal']
            dict_image_paths_concept.update({
                correct_label + '_' + predicted_label: {
                    'image_paths': list_image_paths,
                    'probs': list_probs,
                    'diagonal': diagonal
                }
            }
            )

        self.labels = labels

        return dict_image_paths_concept

    def save_probabilities_labels(self, id, save_path):
        """

        Args:
            id: Name of the file to save (e.g. model id)
            save_path: Folder where to save them

        Returns:

        """
        if self.probabilities is not None and self.labels is not None:
            utils.save_numpy(id + '_probabilities', save_path, self.probabilities)
            utils.save_numpy(id + '_labels', save_path, self.labels)

    def plot_confusion_matrix(self, confusion_matrix, concept_labels=None, save_path=None, show_text=True):
        """

        Args:
            confusion_matrix: Confusion matrix from results
            concept_labels: List containing the class labels
            save_path: If path specified save confusion matrix there
            show_text: If True show text in the confusion matrix

        Returns: Shows the confusion matrix in the screen

        """

        if concept_labels is None:
            concept_labels = utils.get_concept_items(self.concepts, key='label')

        visualizer.plot_confusion_matrix(confusion_matrix, concepts=concept_labels, save_path=save_path,
                                         show_labels=show_text, show_counts=show_text)

    def show_threshold_impact(self, probabilities, labels, type='probability', threshold=None):
        """
        Interactive Plot showing the effect of the threshold
        Args:
            probabilities: Probabilities given by the model [n_samples, n_classes]
            labels: Ground truth labels (categorical)
            type: 'Probability' or 'entropy' for a threshold on network top-1 prob or uncertainty in all predictions
            threshold: Custom threshold

        Returns: The index of the images with error or correct per every threshold, and arrays with the percentage.

        """

        # Get Error Indices, Number of Correct Predictions, Number of Error Predictions per Threshold
        if type == 'probability':
            threshold = threshold or np.arange(0, 1.01, 0.01)
            correct_ind, errors_ind, correct, errors = metrics.get_top1_probability_stats(probabilities,
                                                                                          labels,
                                                                                          threshold, verbose=0)
            n_total_errors = errors[0]
            n_total_correct = correct[0]

        elif type == 'entropy':
            threshold = threshold or np.arange(0, log(probabilities.shape[1], 2) + 0.01, 0.01)
            correct_ind, errors_ind, correct, errors = metrics.get_top1_entropy_stats(probabilities,
                                                                                      labels,
                                                                                      threshold, verbose=0)
            n_total_errors = errors[-1]
            n_total_correct = correct[-1]

        errors = (n_total_errors - errors) / n_total_errors * 100
        correct = correct / n_total_correct * 100

        visualizer.plot_threshold(threshold, correct, errors, title='Top-1 Probability Threshold Tuning')

        return correct_ind, errors_ind, correct, errors

    @staticmethod
    def plot_images(image_paths, n_images=None, title='', n_cols=5, image_res=(20, 20), save_name=None):
        # Works better defining a number of images between 5 and 30 at a time
        """

        Args:
            image_paths: List with image_paths
            n_images: Number of images to show
            title: Title for the plot
            n_cols: Number of columns to split the data
            image_res: Plot image resolution
            save_name: If specified, will save the plot in save_name path

        Returns: Plots images in the screen

        """

        image_paths = np.array(image_paths)
        if n_images is None:
            n_images = image_paths.shape[0]

        visualizer.plot_images(image_paths, n_images, title, None, n_cols, image_res, save_name)

    def plot_probability_histogram(self, bins=200):
        """

        Args:
            bins: Number of histogram bins

        Returns:

        """
        if self.probabilities is None:
            raise ValueError('There are not computed probabilities. Please run an evaluation first.')

        correct, errors = metrics.get_correct_errors_indices(self.probabilities, self.labels, k=1, verbose=0)
        probs_top = np.max(self.probabilities, axis=1)

        probs_errors = probs_top[errors[0]]
        probs_correct = probs_top[correct[0]]

        visualizer.plot_histogram_probabilities(probs_correct, probs_errors, 'Probability Histogram', bins)

    def plot_most_confident(self, mode='errors', title='', n_cols=5, n_images=None, image_res=(20, 20), save_name=None):
        """
            Plots most confident errors or correct detections
            Args:
                mode: Two modes, "correct" and "error" are supported
                title: Title of the Plot
                n_cols: Number of columns
                n_images: Number of images to show
                image_res: Plot image resolution
                save_name: If specified, will save the plot in save_name path

            Returns: Sorted image paths with corresponding probabilities

        """
        if self.probabilities is None:
            raise ValueError('There are not computed probabilities. Please run an evaluation first.')

        correct, errors = metrics.get_correct_errors_indices(self.probabilities, self.labels, k=1, verbose=0)
        probs_top = np.max(self.probabilities, axis=1)

        if mode == 'errors':
            probs = probs_top[errors[0]]
        elif mode == 'correct':
            probs = probs_top[correct[0]]
        else:
            raise ValueError('Incorrect mode. Supported modes are "errors" and "correct"')

        image_paths = np.array(self.image_paths)
        index_max = np.argsort(probs)[::-1]
        image_paths = image_paths[index_max]

        if n_images is None:
            n_images = min(len(image_paths), 20)

        subtitles = ['Prob=' + str(prob)[0:5] for prob in probs[index_max]][0:n_images]

        visualizer.plot_images(image_paths, n_images, title, subtitles[0:n_images], n_cols, image_res, save_name)

        return image_paths, probs[index_max]

    def plot_confidence_interval(self, mode='accuracy', confidence_value=0.95,
                                 probability_interval=np.arange(0, 1.0, 0.01)):
        """
        Computes and plot the confidence interval for a given mode. It uses a confidence value for a given success and
        failure values following a binomial distribution using the gaussian approximation.
        Args:
            mode: Two modes, "accuracy" and "error" are supported
            confidence_value:  Percentage of confidence. Values accepted are 0.9, 0.95, 0.98, 0.99 or 90, 95, 98, 99
            probability_interval: Probabilities to compare with.

        Returns: Mean, lower and upper bounds for each probability. Plot the graph.

        """
        if self.probabilities is None:
            raise ValueError('There are not computed probabilities. Please run an evaluation first.')

        correct, errors = metrics.get_correct_errors_indices(self.probabilities, self.labels, k=1, verbose=0)
        probs_correct = np.max(self.probabilities, axis=1)[correct[0]]
        probs_error = np.max(self.probabilities, axis=1)[errors[0]]

        if mode == 'accuracy':
            title = 'Accuracy Confidence Interval'
            mean, lower_bound, upper_bound = \
                metrics.compute_confidence_interval_binomial(probs_correct, probs_error,
                                                             confidence_value, probability_interval)
        elif mode == 'error':
            title = 'Error Confidence Interval'
            mean, lower_bound, upper_bound = \
                metrics.compute_confidence_interval_binomial(probs_error, probs_correct,
                                                             confidence_value, probability_interval)
        else:
            raise ValueError('Incorrect mode. Modes available are "accuracy" or "error".')

        visualizer.plot_confidence_interval(probability_interval, mean, lower_bound, upper_bound, title=title)
        return mean, lower_bound, upper_bound

    def compute_confidence_prediction_distribution(self, verbose=1):
        """
        Compute the mean value of the probability assigned to predictions, or how confident is the classifier
        Args:
            verbose: If True, show text

        Returns: The mean value of the probability assigned to predictions [top-1, ..., top-k] k = n_classes

        """
        if self.probabilities is None:
            raise ValueError('probabilities value is None, please run an evaluation first')
        return metrics.compute_confidence_prediction_distribution(self.probabilities, verbose)

    def compute_uncertainty_distribution(self, verbose=1):
        """
        Compute how the uncertainty is distributed
        Args:
            verbose: Show text

        Returns: The uncertainty measurement per each sample

        """
        if self.probabilities is None:
            raise ValueError('probabilities value is None, please run an evaluation first')
        return metrics.uncertainty_distribution(self.probabilities, self.combination_mode, verbose)

    def plot_top_k_sensitivity_by_concept(self):
        if self.results is None:
            raise ValueError('results parameter is None, please run an evaluation first')
        concepts = utils.get_concept_items(self.concepts, key='label')
        metrics = [item['metrics']['sensitivity'] for item in self.results['individual']]
        visualizer.plot_concept_metrics(concepts, metrics, 'Top-k', 'Sensitivity')

    def plot_top_k_accuracy(self):
        if self.results is None:
            raise ValueError('results parameter is None, please run an evaluation first')
        metrics = self.results['average']['accuracy']
        visualizer.plot_concept_metrics(['all'], [metrics], 'Top-k', 'Accuracy')

    def show_results(self, mode='average', round_decimals=3, show_id=True):
        """
        Args:
            mode: Mode of results. "average" will show the average metrics while "individual" will show metrics by class
            round_decimals: Decimal position to round the numbers.
            show_id: Show id in the first column.

        Returns: Pandas dataframe with results.

        """
        if self.results is None:
            raise ValueError('results parameter is None, please run an evaluation first')

        return utils.results_to_dataframe(self.results, self.id, mode, round_decimals, show_id)

    def save_results(self, id, csv_path, mode='average', round_decimals=3, show_id=True):
        """

        Args:
            id: Name of the results evaluation
            csv_path: If specified, results will be saved on that location
            mode: Mode of results. "average" will show the average metrics while "individual" will show metrics by class
            round_decimals: Decimal position to round the numbers.
            show_id: Show id in the first column.

        Returns: Nothing. Saves Pandas dataframe on csv_path specified.

        """
        if self.results is None:
            raise ValueError('results parameter is None, please run an evaluation first')

        return utils.save_results(self.results, id, csv_path, mode, round_decimals, show_id)

    def get_sensitivity_per_samples(self, csv_path=None, round_decimals=4, top_k=1):
        """

        Args:
            id: Name of the results evaluation
            csv_path: If specified, results will be saved on that location
            mode: Mode of results. "average" will show the average metrics while "individual" will show metrics by class
            round_decimals: Decimal position to round the numbers.
            show_id: Show id in the first column.

        Returns: Pandas dataframe of results.

        """
        if self.results is None:
            raise ValueError('`self.results` is not computed, please run `self.evaluate()` first')

        results_classes = self.show_results('individual', round_decimals=round_decimals)

        if 0 < top_k <= self.top_k:
            sensitivity_key = 'sensitivity_top_' + str(top_k) if self.top_k > 1 else 'sensitivity'
        else:
            raise ValueError('`top_k` value should be higher than 0 and less than the top_k used in the evaluation.')

        mode = 'n_samples'

        results_classes = results_classes[results_classes.columns.intersection(['class', sensitivity_key, mode])]
        results_classes['% samples'] = results_classes['n_samples'] / results_classes['n_samples'].sum() * 100
        if csv_path is not None:
            results_classes.to_csv(csv_path)

        return results_classes

    def plot_sensitivity_per_samples(self, csv_path=None, round_decimals=4, percentage=True,
                                     n_samples=None, title='Sensitivity per % of Samples',
                                     top_k=1):
        """
        Args:
            csv_path: If specified, results will be saved on that location
            round_decimals: Decimal position to round the numbers.
            percentage: True will show percentage of samples
            n_samples: Will overwrite the samples in the results. Useful to use the training samples counts.
            title: Title of the plot.
            top_k: Top-k sensitivity to select from.

        Returns: Pandas dataframe of results.

        """
        if self.results is None:
            raise ValueError('`self.results` is not computed, please run `self.evaluate()` first')

        if 0 < top_k <= self.top_k:
            sensitivity_key = 'sensitivity_top_' + str(top_k) if self.top_k > 1 else 'sensitivity'
        else:
            raise ValueError('`top_k` value should be higher than 0 and less than the top_k used in the evaluation.')

        results_classes = self.get_sensitivity_per_samples(csv_path, round_decimals, top_k)

        mode = '% samples' if percentage else 'n_samples'

        if n_samples is not None and isinstance(n_samples, list):
            results_classes['n_samples'] = n_samples
            results_classes['% samples'] = results_classes['n_samples'] / results_classes['n_samples'].sum() * 100
        results_classes = results_classes.sort_values(by=sensitivity_key).reset_index(drop=True)

        visualizer.sensitivity_scatter_plot(results_classes[mode],
                                            results_classes[sensitivity_key],
                                            results_classes['class'],
                                            mode,
                                            'Top-{} Sensitivity'.format(str(top_k)),
                                            title)
        return results_classes

    def get_keys_confusion_matrix_errors(self, labels=None, concept_labels=None):
        """

        Args:
            labels: Ground truth labels (categorical), if only prediction was made it should be given
            concept_labels: List with class names (by default last evaluation),
                            if only prediction was made it should be given

        Returns: Keys for the confusion matrix dictionary paths, error counts for each confusion matrix tile

        """
        if labels is None:
            labels = self.labels

        if concept_labels is None:
            if self.concept_labels is None:
                concept_labels = utils.get_concept_items(self.concepts, key='label')
            else:
                concept_labels = self.concept_labels

        dict_paths = self.get_image_paths_by_prediction(self.probabilities, labels, concept_labels)
        keys, counts = [], []
        for key in dict_paths.keys():
            if not dict_paths[key]['diagonal']:
                counts.append(len(dict_paths[key]['image_paths']))
                keys.append(key)

        idx = np.argsort(counts)[::-1]
        keys, counts = np.array(keys), np.array(counts)
        sorted_keys, sorted_counts = keys[idx], counts[idx]

        self.labels = labels
        self.concept_labels = concept_labels

        return sorted_keys, sorted_counts

    def get_errors_confusion_matrix_df(self, labels=None, concept_labels=None, filter_zeros=True, save_path=None):
        """

        Args:
            labels: Ground truth labels (categorical), if only prediction was made it should be given
            concept_labels: List with class names (by default last evaluation),
                            if only prediction was made it should be given
            filter_zeros: If True only save positions with errors
            save_path: Save csv path

        Returns: Dataframe with error counts for each confusion matrix tile

        """
        sorted_keys, sorted_counts = self.get_keys_confusion_matrix_errors(labels, concept_labels)
        dataframe = pd.DataFrame({'matrix_square': sorted_keys, 'count': sorted_counts})

        if filter_zeros:
            dataframe = dataframe[dataframe['count'] != 0]

        if save_path is not None:
            dataframe.to_csv(save_path)

        return dataframe
