import os
import copy
import json
import numpy as np
import pandas as pd
import studio.evaluation.keras.utils as utils

from keras.utils.generic_utils import Progbar
from studio.evaluation.keras.evaluators import Evaluator


class CNNEvaluator(Evaluator):
    OPTIONS = {
        'concepts': {'type': list, 'default': None},
        'config_file_yaml': {'type': list, 'default': None},
        'model_map_json': {'type': None, 'default': None},
        'threshold': {'type': dict, 'default': None},
        'pass_through': {'type': list, 'default': None},
        'squash_classes': {'type': list, 'default': None},
        'ensemble_models_dir': {'type': None, 'default': None},
        'model_path': {'type': None, 'default': None},
        'custom_objects': {'type': None, 'default': None},
        'report_dir': {'type': str, 'default': None},
        'combination_mode': {'type': str, 'default': 'arithmetic'},
        'id': {'type': str, 'default': None},
        'concept_dictionary_path': {'type': str, 'default': None},
        'batch_size': {'type': int, 'default': 1},
        'verbose': {'type': int, 'default': 0},
    }

    def __init__(self, **options):
        """Evaluate the CNNs.

        Args:
            concepts: A list of dictionaries indicating the labels and IDs.
                concepts = [{'label': 'benign', 'id': 'benign'}, {'label': 'malign', 'id': 'malign'}]

        """

        # Be able to load Keras_applications models by default
        self.custom_objects = utils.create_default_custom_objects()

        if 'config_file_yaml' in options.keys() and options['config_file_yaml'] is not None:
            self.options = utils.load_yaml_file(options['config_file_yaml'])
        else:
            self.options = options

        for key, option in self.OPTIONS.items():
            if key not in self.options and 'default' not in option:
                raise ValueError('missing required option: %s' % (key,))
            value = self.options.get(key, copy.copy(option.get('default')))
            if key == 'custom_objects':
                self.update_custom_objects(value)
            elif key == 'concepts':
                if value is not None:
                    self.concepts = value
                elif 'model_map_json' in self.options and self.options['model_map_json'] is not None:
                    pass
                else:
                    self.concepts = None
            elif key == 'model_map_json' and value is not None:
                with open(self.options['model_map_json']) as json_file:
                    data = json.load(json_file)
                self.concepts = [{'id': row['condition_id'],
                                  'label': row['diagnosis_name'],
                                  'diagnosis_id': row['diagnosis_id']} for i, row in enumerate(data)]
                # Obtain labels to show on the metrics results
                self.concept_labels = utils.get_concept_items(self.concepts, key='label')
            elif key == 'combination_mode':
                self.set_combination_mode(value)
            elif key == 'concept_dictionary_path' and self.options.get('concept_dictionary_path') is not None:
                self.concept_dictionary = utils.read_dictionary(value)
            else:
                setattr(self, key, value)
            if key == 'id' and self.options.get('model_path') is not None:
                if value is None:
                    self.id = os.path.basename(self.options.get('model_path'))

        extra_options = set(self.options.keys()) - set(self.OPTIONS.keys())
        if len(extra_options) > 0:
            raise ValueError('unsupported options given: %s' % (', '.join(extra_options),))

        self.results = None
        self.models = []
        self.model_specs = []
        self.probabilities = None
        self.labels = None

        if self.model_path is not None:
            self.add_model(model_path=self.model_path)
        elif self.ensemble_models_dir is not None:
            self.add_model_ensemble(models_dir=self.ensemble_models_dir)

        # else:
        #     raise ValueError('No model information was given')

    def update_custom_objects(self, custom_objects):
        if custom_objects is not None and isinstance(custom_objects, dict):
            for key, value in custom_objects.items():
                self.custom_objects.update({key: value})

    def add_model(self, model_path, specs_path=None, custom_objects=None):
        self.update_custom_objects(custom_objects)
        model, model_spec = utils.load_model(model_path=model_path, specs_path=specs_path,
                                             custom_objects=self.custom_objects)
        self.models.append(model)
        self.model_specs.append(model_spec)

    def add_model_ensemble(self, models_dir, custom_objects=None):
        self.update_custom_objects(custom_objects)
        models, model_specs = utils.load_multi_model(models_dir=models_dir, custom_objects=self.custom_objects)
        for i, model in enumerate(models):
            self.models.append(model)
            self.model_specs.append(model_specs[i])

    def remove_model(self, model_index):
        self.models.pop(model_index)
        self.model_specs.pop(model_index)

    def set_combination_mode(self, mode):
        modes = ['arithmetic', 'geometric', 'maximum', 'harmonic', None]
        if mode in modes:
            self.combination_mode = mode
        else:
            raise ValueError('Error: invalid option for `combination_mode` ' + str(mode))

    def set_concepts(self, concepts):
        for concept_dict in concepts:
            if 'label' not in concept_dict.keys() and 'id' not in concept_dict.keys():
                raise ValueError('Incorrect format for concepts list. It must contain the fields `id` and `label`')
        self.concepts = concepts

    @staticmethod
    def _get_complete_image_paths(data_dir, filenames):
        image_paths = []
        for filename in filenames:
            image_paths.append(os.path.join(data_dir, filename))
        return image_paths

    def evaluate(self,
                 data_dir,
                 dataframe_path=None,
                 top_k=1,
                 filter_indices=None,
                 confusion_matrix=False,
                 custom_crop=True,
                 data_augmentation=None,
                 save_confusion_matrix_path=None,
                 show_confusion_matrix_text=True,
                 interpolation='nearest',
                 validate_filenames=False):
        """Evaluate the model(s) performance.

        Args:
            data_dir: String indicating the directory of images.
            dataframe_path: String indicating the path to the data manifest containing the images information
            top_k: An integer specifying the k most probable predicted labels,
                where if the true label occurs within the k predicted labels,
                the prediction is considered correct.
            filter_indices: If given take only the predictions corresponding to that indices to compute metrics.
            confusion_matrix: If True, show the confusion matrix.
            custom_crop: If True, data generator will crop images when crop coordinates exist for the image.
                Note that this cropping will be done before the `data_augmentation`.
            data_augmentation: A dictionary of augmentations to apply to the image.
                If N augmentations are specified, then the un-augmented image will
                have a single augmentation applied to it, and N augmented images will
                be passed through the CNN, with the resulting probabilities aggregated.
                `data_augmentation` is a dictionary with 3 possible keys:
                - 'scale_sizes': 'default'
                    (4 scales similar to Going Deeper with Convolutions work) or a list of sizes.
                    Each scaled image then will be cropped into three square parts.
                - 'transforms': list of transforms to independently apply to the image.
                    The currently supported transforms are:
                    'horizontal_flip', 'vertical_flip', 'rotate_90', 'rotate_180', 'rotate_270'.
                - 'crop_original': 'center_crop' applies a center crop to the image
                    prior to performing the augmentations.
            save_confusion_matrix_path: File name and path where to save the confusion matrix.
            show_confusion_matrix_text: If False, will hide the text in the confusion matrix.
            interpolation: String indicating the interpolation parameter for the data generator.
            validate_filenames: If True, images with invalid filename extensions will be ignored.

        Returns: A dictionary of the computed metrics between the predicted probabilities and ground truth labels.

        """

        # Needed since used in evaluator.get_sensitivity_per_samples()
        self.top_k = top_k

        self.dataframe = None

        if dataframe_path is not None:
            self.dataframe = pd.read_json(dataframe_path)
            if self.concepts is None:
                self.concepts = [{'id': 'class_' + str(i), 'label': 'class_' + str(i)}
                                 for i in range(len(self.dataframe['class_probabilities'][0]))]
        else:
            # Create dictionary containing class names from folder
            if self.concepts is None:
                self.concepts = utils.get_default_concepts(data_dir)

        # Obtain labels to show on the metrics results
        self.concept_labels = utils.get_concept_items(self.concepts, key='label')

        # Create Keras image generator and obtain probabilities
        probabilities, self.labels_categorical = self._compute_probabilities_generator(
            data_dir=data_dir,
            dataframe=self.dataframe,
            custom_crop=custom_crop,
            data_augmentation=data_augmentation,
            interpolation=interpolation,
            validate_filenames=validate_filenames,
        )

        # Collapse probabilities, obtain 1D label array
        self.probabilities = utils.combine_probabilities(probabilities, self.combination_mode)

        if hasattr(self, 'concept_dictionary'):
            if utils.compare_group_test_concepts(self.concept_labels, self.concept_dictionary) \
                    and utils.check_concept_unique(self.concept_dictionary):

                self.probabilities = self._compute_inference_probabilities(self.probabilities)
            else:
                # Should never be here, but added in case the `utils` function one day fails.
                raise ValueError("Error: Invalid `concept_dictionary`.")

        self.labels = self.labels_categorical.argmax(axis=1)

        # Compute metrics
        self.results = self.get_metrics(probabilities=self.probabilities, labels=self.labels,
                                        concepts=self.concepts, top_k=top_k,
                                        filter_indices=filter_indices,
                                        confusion_matrix=confusion_matrix,
                                        save_confusion_matrix_path=save_confusion_matrix_path,
                                        show_confusion_matrix_text=show_confusion_matrix_text)

        return self.results

    def _compute_probabilities_generator(self,
                                         data_dir=None,
                                         dataframe=None,
                                         custom_crop=True,
                                         data_augmentation=None,
                                         interpolation='nearest',
                                         validate_filenames=False):
        """

        Args:
            data_dir:  Data directory to load the images from
            custom_crop: If True, data generator will crop images.
            data_augmentation: Data augmentation dictionary
            interpolation: String indicating the interpolation parameter for the data generator.
            validate_filenames: If True, images with invalid filename extensions will be ignored.

        Returns: Probabilities, ground truth labels of predictions

        """

        probabilities = []
        if len(self.models) < 1:
            raise ValueError('No models found, please add a valid Keras model first')
        else:
            for i, model in enumerate(self.models):
                print('Making predictions from model ', str(i))
                generator, labels = utils.create_data_generator(data_dir,
                                                                dataframe=dataframe,
                                                                batch_size=self.batch_size,
                                                                model_spec=self.model_specs[i],
                                                                custom_crop=custom_crop,
                                                                data_augmentation=data_augmentation,
                                                                interpolation=interpolation,
                                                                validate_filenames=validate_filenames)
                if data_augmentation is None:
                    # N_batches + 1 to gather all the images + collect without repetition [0:n_samples]
                    probabilities.append(model.predict_generator(generator=generator,
                                                                 steps=(generator.samples // self.batch_size) + 1,
                                                                 workers=1,
                                                                 verbose=1)[0:generator.samples])
                else:
                    print('Averaging probabilities of %i different outputs at sizes: %s with transforms: %s'
                          % (generator.n_crops, generator.scale_sizes, generator.transforms))
                    steps = generator.samples
                    probabilities_model = []
                    for k, batch in enumerate(generator):
                        if k == steps:
                            break
                        progbar = Progbar(steps)
                        progbar.add(k + 1)
                        probs = model.predict(batch[0][0], batch_size=self.batch_size)
                        probabilities_model.append(np.mean(probs, axis=0))
                    probabilities.append(probabilities_model)

            self.generator = generator
            self.num_classes = generator.num_classes
            self.image_paths = self._get_complete_image_paths(data_dir, generator.filenames)

            probabilities = np.array(probabilities)

            return probabilities, labels

    def _compute_inference_probabilities(self, probabilities):
        """
        Args:
            probabilities: Class inference probabilities with shape [samples, inferred_classes].

        Returns: The class probability inference based on key "group" in concept_dictionary

        """

        self.group_id_dict = {}
        for concept in self.concept_dictionary:
            if concept['group'] in self.group_id_dict.keys():
                self.group_id_dict[concept['group']].append(concept['class_index'])
            else:
                self.group_id_dict[concept['group']] = [concept['class_index']]

        inference_probabilities = np.zeros((len(probabilities), len(self.group_id_dict)))

        for idx, key in enumerate(self.group_id_dict.keys()):
            inference_probabilities[:, idx] = np.sum(probabilities[:, self.group_id_dict[key]], axis=1)

        return inference_probabilities

    def predict(self, data_dir, verbose=True, output_all=False):
        """
        Computes a forward pass through the CNN based on options specified and returns probabilities
        Args:
            data_dir: If input is a folder run _predict_folder, if single image run _predict_image() and if it is a
            list of images will run _predict_list()
            verbose: If True will print models output
            output_all: If True instead of probabilitites, it will return masks, concepts and another parameters.
            Mainly used for Sequential Evaluations

        Returns: Or the probabilities computed or the gate prediction information

        """

        self.data_dir = data_dir

        # Compute model probabilities
        if isinstance(self.data_dir, list) or isinstance(self.data_dir, np.ndarray):
            probabilities = self._predict_image_list(data_dir, verbose=verbose)
        elif os.path.isdir(self.data_dir):
            image_list = [os.path.join(self.data_dir, image_path) for image_path in sorted(os.listdir(self.data_dir))]
            probabilities = self._predict_image_list(image_list, verbose=verbose)
        elif self.data_dir.endswith(".png") or self.data_dir.endswith(".jpeg") or self.data_dir.endswith(".jpg"):
            probabilities = self._predict_image(self.data_dir, verbose=verbose)
        else:
            raise ValueError('Wrong data format inputted, please input a valid directory, list of image paths or '
                             'single image path')

        probabilities = utils.combine_probabilities(probabilities, self.combination_mode)

        # Create output placeholders
        if self.concepts is None:
            n_model_classes = int(self.models[0].output.shape[1])
            self.concepts = ['concept_' + str(n) for n in range(n_model_classes)]

        concepts_evaluator = np.array(self.concepts)
        evaluation_classes_idx = list(range(len(self.concepts)))
        n_samples = probabilities.shape[0]
        pass_through_mask = np.ones(n_samples)
        threshold_mask = np.zeros(n_samples)

        # Squash probabilities if requested
        if 'squash_classes' in self.options.keys() and utils.check_squash_classes(
                len(evaluation_classes_idx), self.options['squash_classes']):
            print("Squashing classes")

            concepts_evaluator_squashed = []
            evaluation_classes_idx = []
            probabilities_squashed = np.zeros((probabilities.shape[0], len(self.options['squash_classes'])))
            # It will merge the classes probs into one and do the same with the concepts separating them with a '_'
            for i, class_indices in enumerate(self.options['squash_classes']):
                probabilities_squashed[:, i] = np.sum(probabilities[:, class_indices], axis=1)
                ids_aux, labels_aux = [], []
                for item in concepts_evaluator[class_indices]:
                    ids_aux += item['id']
                    labels_aux += item['label']
                concepts_evaluator_squashed.append({'id': ids_aux, 'label': '_'.join(labels_aux)})
                evaluation_classes_idx.append(i)

            concepts_evaluator = concepts_evaluator_squashed
            probabilities = probabilities_squashed

        # Apply thresholds based on trigger classes and probabilities
        if 'threshold' in self.options.keys() and utils.check_threshold_classes(
                len(evaluation_classes_idx), self.options['threshold']):
            print("Applying threshold")
            threshold_mask = np.zeros(shape=n_samples, dtype=bool)
            predictions = np.argmax(probabilities, axis=1)
            threshold_class_idx = self.options['threshold']['threshold_class']
            for i, pred in enumerate(predictions):
                if pred in self.options['threshold']['trigger_classes']:
                    if probabilities[i, threshold_class_idx] >= self.options['threshold']['threshold_prob']:
                        probabilities[i, threshold_class_idx] = 1.0
                        threshold_mask[i] = True

        # Filter predictions on the squashed and thresholded final probabilities
        if 'pass_through' in self.options.keys() and utils.check_pass_through_classes(
                len(evaluation_classes_idx), self.options['pass_through']):
            print('Filtering outputs')
            predictions = np.argmax(probabilities, axis=1)
            pass_through_mask = np.isin(predictions, self.options['pass_through'])

            evaluation_classes_idx = []
            concepts_auxiliar = []
            for j, item in enumerate(concepts_evaluator):
                if j not in self.options['pass_through']:
                    concepts_auxiliar.append(item)
                    evaluation_classes_idx.append(j)
            concepts_evaluator = concepts_auxiliar

        self.probabilities = probabilities

        if output_all:
            return {
                'probabilities': probabilities,
                'evaluation_classes_idx': np.array(evaluation_classes_idx),
                'pass_through_mask': pass_through_mask,
                'concepts_evaluator': np.array(concepts_evaluator),
                'threshold_mask': threshold_mask,
            }
        else:
            return probabilities

    def _predict_image_list(self, image_list, verbose=True):
        """
        Predict the class probabilities of a set of images from a given list.
        Args:
            image_list: List of image paths
            verbose: If True will print models output

        Returns: Probabilities predicted

        """

        self.batch_size = min(len(image_list), self.batch_size)
        image_list = utils.get_valid_images(image_list)
        n_batches, n_remainder = utils.get_n_batches(len(image_list), self.batch_size)

        probabilities = []
        for i, model in enumerate(self.models):
            probabilities_model = []
            # Progress Bar
            if verbose:
                print('Making predictions from model ', str(i))
                progbar = Progbar(n_batches)
                progbar.verbose = verbose
            # Loop over all the batches.
            for batch_idx in np.arange(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                filenames = image_list[start_idx:end_idx]
                # Read images from folder
                images = utils.load_preprocess_image_list(filenames, self.model_specs[i])
                images = np.array(images)
                # Predict
                probabilities_model.append(model.predict(x=images, batch_size=self.batch_size, verbose=False))
                if verbose:
                    progbar.update(batch_idx)
            # Handle the remainder.
            if n_remainder > 0:
                if verbose:
                    print('\nProcessing remainder: %i' % n_remainder)
                filenames = image_list[-n_remainder:]
                # Read images from folder
                images = utils.load_preprocess_image_list(filenames, self.model_specs[i])
                images = np.array(images)
                # Predict
                probabilities_model.append(model.predict(x=images, batch_size=self.batch_size, verbose=False))

            probabilities.append(np.array([item for sublist in probabilities_model for item in sublist]))

        self.probabilities = np.array(probabilities)
        self.image_paths = image_list

        return self.probabilities

    def _predict_image(self, image_path, verbose):
        """
        Predict class probabilities for a single image.
        Args:
            image_path: Path where the image is located
            verbose: If True will print models output

        Returns: Class probabilities for a single image

        """
        probabilities = []
        for i, model in enumerate(self.models):
            # Read image
            image = utils.load_preprocess_image(image_path, self.model_specs[i])
            # Predict
            if verbose:
                print('Making predictions from model ', str(i))
            probabilities.append(model.predict(x=image, batch_size=1, verbose=verbose))

        self.probabilities = np.array(probabilities)
        self.image_paths = [image_path]

        return self.probabilities

    def ensemble_models(self, input_shape, combination_mode='average', ensemble_name='ensemble', model_filename=None):
        ensemble = utils.ensemble_models(self.models, input_shape=input_shape, combination_mode=combination_mode,
                                         ensemble_name=ensemble_name)
        if model_filename is not None:
            ensemble.save(model_filename)
        return ensemble
