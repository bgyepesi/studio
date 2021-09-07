import copy
import numpy as np
import studio.evaluation.keras.utils as utils

from studio.evaluation.keras.evaluators import Evaluator
from studio.evaluation.keras.evaluators import CNNEvaluator


class SequentialCNNEvaluator(Evaluator):
    OPTIONS = {
        'gates_configuration_path': {'type': None},
        'concepts': {'type': list, 'default': None},
        'custom_objects': {'type': None, 'default': None},
        'report_dir': {'type': str, 'default': None},
        'id': {'type': str, 'default': None},
        'batch_size': {'type': int, 'default': 1},
        'verbose': {'type': int, 'default': 0},
    }

    def __init__(self, **options):
        for key, option in self.OPTIONS.items():

            if key not in options and 'default' not in option:
                raise ValueError('missing required option: %s' % (key,))

            value = options.get(key, copy.copy(option.get('default')))
            setattr(self, key, value)

        extra_options = set(options.keys()) - set(self.OPTIONS.keys())
        if len(extra_options) > 0:
            raise ValueError('unsupported options given: %s' % (', '.join(extra_options),))

        self.cnn_evaluators = []

        # Read gates information files
        for gate_config_path in self.gates_configuration_path:
            # Create CNN evaluators objects for each gate and get list of concepts ids (aip ids)
            cnn_evaluator = CNNEvaluator(config_file_yaml=gate_config_path)
            self.cnn_evaluators.append(cnn_evaluator)

    def _forward_gate_probabilities(self):
        """

        Iterate through all the gates, we don't mix predictions with the following gates, so predictions from latter
        gates do not increase performance of former gates (e.g. prediction from former gate is higher than predictions
        from latter gates but it has already been filtered, so we don't take it into account)

        Returns: Probabilities of the system after passing through all the gates

        """
        probabilities = np.zeros((self.n_samples, len(self.concepts_evaluation)))
        class_idx = 0
        pass_through_mask = None

        for gate_info in self.gates_information:

            # First gate
            if pass_through_mask is None:
                evaluation_mask = ~ gate_info['pass_through_mask']
                probabilities[evaluation_mask, class_idx:class_idx + len(gate_info['concepts_evaluator'])] = \
                    gate_info['probabilities'][np.ix_(evaluation_mask, gate_info['evaluation_classes_idx'])]
                pass_through_mask = gate_info['pass_through_mask']

            # All following gates
            else:
                probabilities[pass_through_mask, class_idx:class_idx + len(gate_info['concepts_evaluator'])] \
                    = gate_info['probabilities'][:, gate_info['evaluation_classes_idx']]

                pass_through_mask = gate_info['pass_through_mask']

            class_idx = class_idx + len(gate_info['concepts_evaluator'])
        return probabilities

    def evaluate(self, image_paths, labels, top_k=1, concepts_mode='label', data_augmentation=None,
                 confusion_matrix=False, save_confusion_matrix_path=None, show_confusion_matrix_text=True):
        """

        Args:
            image_paths: List or numpy array containing a list of image paths
            labels: List or numpy array containing a list of labels
            top_k: The top-k predictions to consider. E.g. top_k = 5 is top-5 preds
            concepts_mode: Whether to show 'id' or 'label' of concepts
            data_augmentation: Data augmentation dictionary
            confusion_matrix: True/False whether to show the confusion matrix
            save_confusion_matrix_path: If path specified save confusion matrix there
            show_confusion_matrix_text: If False, will hide the text in the confusion matrix

        Returns: Probabilities computed and ground truth labels associated. Stores results in self.results

        """

        self.gates_information = []
        self.image_paths = np.array(image_paths)
        self.original_labels = np.array(labels)
        self.concepts_evaluation = np.array([])
        self.n_samples = len(image_paths)

        pass_through_mask = np.ones(self.n_samples, dtype=bool)
        threshold_mask = np.zeros(self.n_samples, dtype=bool)

        # For assigning correct labels
        class_offset = 0
        previous_labels = None

        # Iterate over each gate and compute probabilities for the samples that meet the gate specifications
        for k, cnn_evaluator in enumerate(self.cnn_evaluators):
            concepts_evaluator, gate_config = cnn_evaluator.concepts, cnn_evaluator.options
            n_classes_output = len(cnn_evaluator.concepts)
            if len(image_paths) > 1:

                # Compute forward pass through models, extract probabilities and masks
                gate_info = cnn_evaluator.predict(data_dir=image_paths, output_all=True)

                # Match gate labels with ground truth
                gate_labels = []
                no_label_found = True

                for label_id, label_group in enumerate(self.original_labels):
                    for label in label_group:
                        for ind, concept in enumerate(concepts_evaluator):
                            if label in concept['id']:
                                no_label_found = False
                                gate_labels.append(class_offset + ind)
                    # If there was a prediction error and there is no label, we pick previous label
                    if no_label_found and previous_labels is not None:
                        gate_labels.append(previous_labels[label_id])
                    no_label_found = True

                gate_labels = np.array(gate_labels)

                # Squash probabilities if requested
                if 'squash_classes' in gate_config.keys() and utils.check_squash_classes(
                        n_classes_output, gate_config['squash_classes']):
                    print("Squashing labels... ")
                    for i, class_indices in enumerate(gate_config['squash_classes']):
                        # Create a binary mask and squash labels too
                        gate_labels[np.isin(gate_labels, class_indices)] = i
                    n_classes_output -= n_classes_output - len(gate_config['squash_classes'])

                # Apply thresholds based on trigger classes and probabilities
                if 'threshold' in gate_config.keys() and utils.check_threshold_classes(
                        n_classes_output, gate_config['threshold']):
                    print("Updating threshold mask...")
                    # Update mask of samples affected
                    new_pass_idx = 0
                    for m in range(len(threshold_mask)):
                        if not threshold_mask[m]:
                            threshold_mask[m] = gate_info['threshold_mask'][new_pass_idx]
                            new_pass_idx += 1

                # Filter predictions on the squashed and thresholded final probabilities
                if 'pass_through' in gate_config.keys() and utils.check_pass_through_classes(
                        n_classes_output, gate_config['pass_through']):
                    print('Updating pass through mask...')
                    # Update previous mask
                    new_pass_idx = 0
                    for m in range(len(pass_through_mask)):
                        if pass_through_mask[m]:
                            pass_through_mask[m] = gate_info['pass_through_mask'][new_pass_idx]
                            new_pass_idx += 1

                self.concepts_evaluation = np.concatenate((self.concepts_evaluation, gate_info['concepts_evaluator']))

                # Next iteration parametres
                image_paths = self.image_paths[pass_through_mask]
                class_offset += len(gate_info['concepts_evaluator'])
                previous_labels = gate_labels

                # Save gate information
                gate_info.update({
                    'labels': gate_labels,
                    'pass_through_mask': pass_through_mask,
                    'threshold_mask': threshold_mask
                })
                self.gates_information.append(gate_info)
            else:
                print('All images have been already filtered before the %i gate' % (k + 1))

        # Compute whole system metrics
        if len(self.gates_information) > 1:
            # Create placeholders to store probabilities and labels
            self.probabilities = self._forward_gate_probabilities()
            # Labels have been updated each gate iteration, last iteration contains our evaluation labels
            self.labels = self.gates_information[-1]['labels']
        else:
            print('There are only probabilities for the first gate')
            self.probabilities = self.gates_information[0]['probabilities']
            self.labels = self.gates_information[0]['labels']

        self.concepts = self.concepts_evaluation

        # Compute metrics
        self.results = self.get_metrics(probabilities=self.probabilities, labels=self.labels,
                                        concepts=self.concepts, top_k=top_k,
                                        confusion_matrix=confusion_matrix,
                                        save_confusion_matrix_path=save_confusion_matrix_path,
                                        show_confusion_matrix_text=show_confusion_matrix_text)

        return self.results

    def predict(self, image_paths, data_augmentation=False):
        """

        Args:
            image_paths: List or numpy array containing a list of image paths
            data_augmentation: Data augmentation dictionary

        Returns: Probabilities computed

        """
        self.image_paths = image_paths
        self.gates_information = []
        self.concepts_evaluation = np.array([])
        self.n_samples = len(image_paths)

        pass_through_mask = np.ones(self.n_samples, dtype=bool)
        threshold_mask = np.zeros(self.n_samples, dtype=bool)

        # Iterate over each gate and compute probabilities for the samples that meet the gate specifications
        for k, cnn_evaluator in enumerate(self.cnn_evaluators):
            gate_config = cnn_evaluator.options
            n_classes_output = len(cnn_evaluator.concepts)
            if len(image_paths) > 1:
                # Compute forward pass through models, extract probabilities and masks
                gate_info = cnn_evaluator.predict(data_dir=image_paths, output_all=True)

                # Squash probabilities if requested
                if 'squash_classes' in gate_config.keys() and utils.check_squash_classes(
                        n_classes_output, gate_config['squash_classes']):
                    n_classes_output -= n_classes_output - len(gate_config['squash_classes'])

                # Apply thresholds based on trigger classes and probabilities
                if 'threshold' in gate_config.keys() and utils.check_threshold_classes(
                        n_classes_output, gate_config['threshold']):
                    print("Updating threshold mask...")
                    # Update mask of samples affected
                    new_pass_idx = 0
                    for m in range(len(threshold_mask)):
                        if not threshold_mask[m]:
                            threshold_mask[m] = gate_info['threshold_mask'][new_pass_idx]
                            new_pass_idx += 1

                # Filter predictions on the squashed and thresholded final probabilities
                if 'pass_through' in gate_config.keys() and utils.check_pass_through_classes(
                        n_classes_output, gate_config['pass_through']):
                    print('Updating pass through mask...')
                    # Update previous mask
                    new_pass_idx = 0
                    for m in range(len(pass_through_mask)):
                        if pass_through_mask[m]:
                            pass_through_mask[m] = gate_info['pass_through_mask'][new_pass_idx]
                            new_pass_idx += 1

                self.concepts_evaluation = np.concatenate((self.concepts_evaluation, gate_info['concepts_evaluator']))

                # Next iteration parametres
                image_paths = self.image_paths[pass_through_mask]

                # Save gate information
                gate_info.update({
                    'pass_through_mask': pass_through_mask,
                    'threshold_mask': threshold_mask
                })
                self.gates_information.append(gate_info)
            else:
                print('All images have been already filtered before the %i gate' % (k + 1))

        # Compute whole system metrics
        if len(self.gates_information) > 1:
            self.probabilities = self._forward_gate_probabilities()
        else:
            print('There are only probabilities for the first gate')
            self.probabilities = self.gates_information[0]['probabilities']

        self.concepts = self.concepts_evaluation

        return self.probabilities
