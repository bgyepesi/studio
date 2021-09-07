import os
import csv
import json
import yaml
import itertools
import scipy.stats
import numpy as np
import keras.models
import tensorflow as tf
import pandas as pd

from tensorflow.keras import backend
from tensorflow.keras.layers import average, maximum
from keras.models import Model, Input
from keras_model_specs import ModelSpec
from studio.evaluation.keras.data_generators import EnhancedImageDataGenerator


VALID_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG')


def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    else:
        return numerator / denominator


def round_list(input_list, decimals=7):
    return [round(x, ndigits=decimals) for x in input_list]


def swish(x):
    """Swish activation function.

    # Arguments
        x: Input tensor.

    # Returns
        The Swish activation: `x * sigmoid(x)`.

    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if backend.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return backend.tf.nn.swish(x)
        except AttributeError:
            pass

    return x * backend.sigmoid(x)


def create_default_custom_objects():
    """

    Returns: Default custom objects for Keras models supported in keras_applications

    """
    return {'tf': tf, 'swish': swish}


def load_multi_model(models_dir, custom_objects=None):
    """
    Loads multiple models stored in `models_path`.

    Args:
       models_path: A string indicating the directory were models are stored.
       custom_objects: Dict mapping class names (or function names) of custom (non-Keras) objects to class/functions.

    Returns: List of models, list of model_specs

    """

    models = []
    model_specs = []
    num_models = 0
    model_extensions = ['.h5', '.hdf5']

    for dirpath, dirnames, files in os.walk(models_dir):
        for dir in dirnames:
            files = sorted(os.listdir(os.path.join(dirpath, dir)))
            for filename in files:
                if filename.endswith(tuple(model_extensions)):
                    print('Loading model ', os.path.join(dirpath, dir, filename))
                    model, model_spec = load_model(os.path.join(dirpath, dir, filename), custom_objects=custom_objects)
                    models.append(model)
                    model_specs.append(model_spec)
                    num_models += 1

    print('Models loaded: ', num_models)
    return models, model_specs


def load_model(model_path, specs_path=None, custom_objects=None):
    """

    Args:
        model_dir: Folder containing the model
        specs_path: If specified custom model_specs name, default `model_spec.json`
        custom_objects: Dict mapping class names (or function names) of custom (non-Keras) objects to class/functions.

    Returns: keras model, model_spec object for that model

    """
    model = keras.models.load_model(model_path, custom_objects)
    if specs_path is None:
        model_name = model_path.split('/')[-1]
        specs_path = model_path.replace(model_name, 'model_spec.json')
    with open(specs_path) as f:
        model_spec_json = json.load(f)
        model_spec = ModelSpec(model_spec_json)
    return model, model_spec


def ensemble_models(models, input_shape, combination_mode='average', ensemble_name='ensemble'):
    """

    Args:
        models: List of keras models
        input_shape: Tuple containing input shape in tf format (H, W, C)
        combination_mode: The way probabilities will be joined. We support `average` and `maximum`
        ensemble_name: The name of the model that will be returned

    Returns: A model containing the ensemble of the `models` passed. Same `input_shape` will be used for all of them

    """
    if not len(input_shape) == 3:
        raise ValueError('Incorrect input shape, it should have 3 dimensions (H, W, C)')
    input_shape = Input(input_shape)
    combination_mode_options = ['average', 'maximum']
    # Collect outputs of models in a list

    models_output = []
    for i, model in enumerate(models):
        # Keras needs all the models to be named differently
        model.name = 'model_' + str(i)
        models_output.append(model(input_shape))

    # Computing outputs
    if combination_mode in combination_mode_options:
        if combination_mode == 'average':
            out = average(models_output)
        elif combination_mode == 'maximum':
            out = maximum(models_output)
        # Build model from same input and outputs
        ensemble = Model(inputs=input_shape, outputs=out, name=ensemble_name)
    else:
        raise ValueError('Incorrect combination mode selected, we only allow for `average` or `maximum`')

    return ensemble


def get_default_concepts(data_dir):
    """
    Creates default concepts dictionary from data_dir folder names
    Args:
        data_dir: String indicating the path where the concept folders are

    Returns:
        concepts: Dictionary with 'label' and 'id' equal to each folder name
    """

    if not os.path.exists(data_dir):
        raise ValueError('data_dir path does not exist')

    concepts = []
    for directory in sorted(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, directory)):
            concepts.append({'label': directory, 'id': directory})
    return concepts


def get_dictionary_concepts(model_dictionary):
    """
    Returns concept list from a model dictionary.

    Args:
        model_dictionary: String indicating the path where the model_dictionary json file is.
                        This dictionary must contain 'class_index' and 'class_name' for each class.

    Returns:
        concepts: Dictionary with 'label' and 'id' equal to 'class_name' and 'class_index' for each class.
    """
    if not os.path.isfile(model_dictionary):
        raise ValueError('model_dictionary file does not exist')

    concepts = []
    for model_class in read_dictionary(model_dictionary):
        concepts.append({'label': model_class['class_name'], 'id': model_class['class_index']})
    return concepts


def get_concept_items(concepts, key):
    return [concept[key] for concept in concepts]


def read_dictionary(dictionary_path):
    if os.path.exists(dictionary_path):
        with open(dictionary_path, 'r') as dictionary_file:
            dictionary = json.load(dictionary_file)
    else:
        raise ValueError('Error: invalid dictionary path' + str(dictionary_path))
    return dictionary


def create_training_json(train_dir, output_json_file):
    """
    Checks if evaluation concepts are unique
    Args:
        train_dir: The location where you have the training directory
        output_json_file: The output file name and path e.g.: ./dictionary.json

    Returns:
        True, if there are no repeat concepts, else raises error
    """
    concept_dict = []
    train_concepts = get_default_concepts(train_dir)
    for idx in range(len(train_concepts)):
        concept_dict.append(
            {"class_index": idx, "class_name": train_concepts[idx]["label"], "group": train_concepts[idx]["label"]})
    with open(output_json_file, 'w') as file_obj:
        json.dump(concept_dict, file_obj, indent=4, sort_keys=True)


def check_input_samples(y_probs, y_true):
    """
    Checks if number predicted samples from 'y_probs' is the same as the ground truth samples from 'y_true'
    Args:
        y_probs: A numpy array of the class probabilities.
        y_true: A numpy array of the true class labels (*not* encoded as 1-hot).
    Returns:
        True, if len(y_probs) == len(y_true), otherwise raises error
    """
    if len(y_probs) != len(y_true):
        raise ValueError('The number predicted samples (%i) is different from the ground truth samples (%i)' %
                         (len(y_probs), len(y_true)))
    else:
        return True


def check_top_k_concepts(concepts, top_k):
    """
    Checks if the 'top_k' requested is not higher than the number of 'concepts', or zero.
    Args:
        concepts: A list containing the names of the classes.
        top_k: A number specifying the top-k results to compute. E.g. 2 will compute top-1 and top-2
    Returns:
        True, if len(top_k)>0 && len(top_k)>len(concepts), otherwise raises error
    """
    if top_k <= 0 or top_k > len(concepts):
        raise ValueError('`top_k` value should be between 1 and the total number of concepts (%i)' % len(concepts))
    else:
        return True


def check_concept_unique(concept_dict):
    """
    Checks if evaluation concepts are unique
    Args:
        concept_dict: Dictionary that contains class_id, train_concepts and groups
    Returns:
        True, if there are no repeat concepts, else raises error
    """
    concept_class_name_dict = {}
    for concept_dict_item in concept_dict:
        if concept_dict_item['class_name'] in concept_class_name_dict:
            raise ValueError("Concept has been repeated:", concept_dict_item['class_name'])
        else:
            concept_class_name_dict[concept_dict_item['class_name']] = 1

    return True


def compare_group_test_concepts(test_concepts_list, concept_dict):
    """
    Checks if concept dictionary has the groups as the test concepts
    Args:
        test_concepts_list: List of labels corresponding to the test concepts
        concept_dict: Dictionary that contains class_id, train_concepts and groups
    Returns:
        True, if there are no repeat concepts, else raises error
    """
    concept_group_list = get_concept_items(concept_dict, key="group")

    different_concept_set = set(concept_group_list).symmetric_difference(set(test_concepts_list))
    if len(different_concept_set):
        raise ValueError(
            "The following concepts are not present in either the concept dictionary or among the test classes:",
            list(different_concept_set))

    else:
        return True


def create_data_generator(data_dir,
                          batch_size,
                          model_spec,
                          dataframe=None,
                          custom_crop=False,
                          data_augmentation=None,
                          interpolation='nearest',
                          validate_filenames=False):
    """

    Args:
        data_dir: Root directory where data is stored
        batch_size: Number of images per batch
        model_spec: Model spec
        dataframe: If dataframe given will flow from dataframe, else from directory
        custom_crop: If True, data generator has to crop images according to `crop_col`.
        data_augmentation: Data augmentation dictionary
        validate_filenames: If True, images with invalid filename extensions will be ignored.

    Returns: An Enhanced Keras Image Generator.

    """

    data_generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                custom_crop=custom_crop,
                                                data_augmentation=data_augmentation)
    if dataframe is not None:
        generator = data_generator.flow_from_dataframe(dataframe,
                                                       directory=data_dir,
                                                       batch_size=batch_size,
                                                       x_col="filename",
                                                       y_col="class_probabilities",
                                                       crop_col='crop',
                                                       target_size=model_spec.target_size[:2],
                                                       class_mode='probabilistic',
                                                       shuffle=False,
                                                       interpolation=interpolation,
                                                       validate_filenames=validate_filenames)
    else:
        generator = data_generator.flow_from_directory(data_dir,
                                                       batch_size=batch_size,
                                                       target_size=model_spec.target_size[:2],
                                                       class_mode='categorical',
                                                       interpolation=interpolation,
                                                       shuffle=False)

    print('Input image size: ', model_spec.target_size)

    labels = keras.utils.np_utils.to_categorical(generator.classes, generator.num_classes)

    return generator, labels


def get_valid_images(image_list):
    """

    Args:
        image_list: List of images

    Returns: List of images with valid image format. Prints image_paths that are not valid.

    """
    image_paths = []
    for image_path in image_list:
        if image_path.endswith(VALID_IMAGE_FORMATS):
            image_paths.append(image_path)
        else:
            print('Incorrect image format for %s, skipping' % image_path)
    return image_paths


def load_preprocess_image(image_path, model_spec):
    """

    Args:
        image_path: A string indicating the name and path of the image.
        model_spec: Model Spec object that includes parameters as the load size or the pre-processing function

    Returns: The pre-processed image.

    """

    return model_spec.load_image(image_path)


def load_preprocess_image_list(image_list, model_spec):
    """

    Args:
        image_list: A list of paths to images.
        model_spec: Model Spec object that includes parameters as the load size or the pre-processing function

    Returns: An array of pre-processed images.

    """

    return [load_preprocess_image(file_path, model_spec)[0] for file_path in image_list]


def combine_probabilities(probabilities, combination_mode='arithmetic', ensemble_weights=None):
    """
    Args:
        probabilities: Probabilities given by the ensemble of models
        combination_mode: Combination_mode: 'arithmetic' / 'geometric' / 'harmonic' mean of the predictions or 'maximum'
           probability value
        ensemble_weights: If provided, it will compute the weighted average of each model's probabilities

    Returns: Probabilities combined
    """

    combiners = {
        'arithmetic': np.mean,
        'geometric': scipy.stats.gmean,
        'harmonic': scipy.stats.hmean,
        'maximum': np.amax
    }

    # Probabilities of the ensemble input=[n_models, n_samples, n_classes] --> output=[n_samples, n_classes]

    if ensemble_weights is not None:
        if len(ensemble_weights) != probabilities.shape[0]:
            raise ValueError('Length of weights %d do not coincide with the number of models %d'
                             % (len(ensemble_weights), probabilities.shape[0]))
    # Make sure we have a numpy array
    probabilities = np.array(probabilities)

    # Join probabilities given by an ensemble of models following combination mode
    if probabilities.ndim == 3:
        if probabilities.shape[0] <= 1:
            return probabilities[0]
        else:
            # Combine ensemble probabilities
            if combination_mode not in combiners.keys():
                raise ValueError('Error: invalid option for `combination_mode` ' + str(combination_mode))
            else:
                if ensemble_weights is not None:
                    if np.isclose(np.sum(ensemble_weights), 1):
                        return np.average(probabilities, weights=ensemble_weights, axis=0)
                    else:
                        raise ValueError('The sum of the weights provided (%f) do not aggregate to 1.0'
                                         % (np.sum(ensemble_weights)))
                else:
                    return combiners[combination_mode](probabilities, axis=0)

    elif probabilities.ndim == 2:
        return probabilities
    else:
        raise ValueError('Incorrect shape for `probabilities` array, we accept [n_samples, n_classes] or '
                         '[n_models, n_samples, n_classes]')


def results_to_dataframe(results, id='default_model', mode='average', round_decimals=3, show_id=True, hide_index=False):
    """

    Converts results to pandas to show a nice visualization of the results. Allow saving them to a csv file.

    Args:
        results: Results dictionary provided by the evaluation (evaluator.results)
        id: Name of the results evaluation
        mode: Mode of results. "average" will show the average metrics while "individual" will show metrics by class
        csv_path: If specified, results will be saved on that location
        round_decimals: Decimal position to round the numbers.
        show_id: Show id in the first column.
        hide_index: Show index in the first column.

    Returns: A pandas dataframe with the results and prints a nice visualization

    """

    if mode not in ['average', 'individual']:
        raise ValueError('Results mode must be either "average" or "individual"')

    if mode == 'average':
        df = pd.DataFrame({'model_id': id}, index=range(1))

        for metric in results['average'].keys():
            if metric != 'confusion_matrix':
                if not isinstance(results['average'][metric], list):
                    df[metric] = round(results['average'][metric], round_decimals)
                else:
                    if len(results['average'][metric]) == 1:
                        df[metric] = round(results['average'][metric][0], round_decimals)
                    else:
                        for k in range(len(results['average'][metric])):
                            df[metric + '_top_' + str(k + 1)] = round(results['average'][metric][k], round_decimals)

    if mode == 'individual':
        df = pd.DataFrame()
        metrics = results['individual'][0]['metrics'].keys()
        df['id'] = [result['id'] for result in results['individual']]
        df['class'] = [result['label'] for result in results['individual']]

        for metric in metrics:
            if not isinstance(results['individual'][0]['metrics'][metric], list):
                concept_list = []
                for idx, concept in enumerate(df['class']):
                    concept_list.append(round(results['individual'][idx]['metrics'][metric], round_decimals))
                df[metric] = concept_list
            elif len(results['individual'][0]['metrics'][metric]) == 1:
                concept_list = []
                for idx, concept in enumerate(df['class']):
                    concept_list = round(results['individual'][idx]['metrics'][metric][0], round_decimals)
                df[metric] = concept_list
            else:
                for k in range(len(results['individual'][0]['metrics'][metric])):
                    concept_list = []
                    for idx, concept in enumerate(df['class']):
                        concept_list.append(
                            round(results['individual'][idx]['metrics'][metric][k], round_decimals))
                    df[metric + '_top_' + str(k + 1)] = concept_list
    if not show_id:
        df.drop('id', axis=1, inplace=True)

    if hide_index:
        df = df.style.hide_index()

    return df


def mkdir(path):
    """

    Args:
        path: Path where directory will be created

    Returns: Nothing. Creates directory with the path specified

    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_numpy(id, path, file):
    np.save(os.path.join(path, id + '.npy'), file)


def save_results(results, id, csv_path, mode='average', round_decimals=3, show_id=True):
    """

    Args:
        results: Results dictionary provided by the evaluation (evaluator.results)
        id: Name of the results evaluation
        mode: Mode of results. "average" will show the average metrics while "individual" will show metrics by class
        csv_path: If specified, results will be saved on that location
        round_decimals: Decimal position to round the numbers.
        show_id: Show id in the first column.

    Returns: Nothing. Saves pandas dataframe on csv_path specified.

    """
    df = results_to_dataframe(results, id=id, mode=mode, round_decimals=round_decimals, show_id=show_id)
    mkdir(csv_path)
    df.to_csv(os.path.join(csv_path, id + '_' + mode + '.csv'), float_format='%.' + str(round_decimals) + 'f',
              index=False)


def load_csv_to_dataframe(csv_paths):
    """

    Args:
        csv_paths: Path or list of paths to the csvs

    Returns: A Pandas dataframe containing the csv information

    """
    results_dataframe = []
    if isinstance(csv_paths, list):
        for path in csv_paths:
            results_dataframe.append(pd.read_csv(path))
    elif isinstance(csv_paths, str):
        results_dataframe = pd.read_csv(path)
    else:
        raise ValueError('Incorrect format for `csv_paths`, a list of strings or a single string are expected')
    return results_dataframe


def compute_differential_str(value_reference, value, round_decimals):
    """

    Args:
        value_reference: Reference Value
        value: Value to modify
        round_decimals: Decimal position to round the numbers.

    Returns: A string with the differential between the two values (value - value_reference)

    """
    diff_value = round(value - value_reference, round_decimals)
    if diff_value > 0:
        return ' (+' + str(diff_value) + ')'
    else:
        return ' (' + str(diff_value) + ')'


def results_differential(dataframes, mode='average', round_decimals=4, save_csv_path=None):
    """

    Args:
        dataframes: List of results dataframes. The first one will be considered the reference.
        mode: Mode of results. "average" will show the average metrics while "individual" will show metrics by class
        round_decimals: Decimal position to round the numbers.
        save_csv_path: Path to save the resulting dataframe with the differential information.

    Returns: Modified dataframe with the differential information.

    """
    if len(dataframes) < 2:
        raise ValueError('The number of dataframes should be higher than 1')

    if mode == 'average':
        skip_values = ['id', 'number_of_samples', 'number_of_classes']
        reference_dataframe = dataframes.pop(0)
        differential_dataframe = reference_dataframe.copy()
        for dataframe in dataframes:
            for name, values in dataframe.iteritems():
                if name not in skip_values:
                    diff_str = compute_differential_str(reference_dataframe[name][0], dataframe[name][0],
                                                        round_decimals)
                    dataframe[name] = str(dataframe[name][0]) + diff_str
            differential_dataframe = pd.concat((differential_dataframe, dataframe), ignore_index=True)

    elif mode == 'individual':
        skip_values = ['id', '% of samples', 'class']
        n_evaluations = len(dataframes)
        differential_dataframe = pd.concat(dataframes, ignore_index=True)
        differential_dataframe = differential_dataframe.rename_axis('index').sort_values(by=['class', 'index'],
                                                                                         ascending=[True, True])
        differential_dataframe = differential_dataframe.reset_index(drop=True)
        reference_index = 0
        for index, row in differential_dataframe.iterrows():
            if index % n_evaluations == 0:
                reference_index = index
            else:
                reference_row = differential_dataframe.iloc[reference_index]
                for name in list(differential_dataframe.columns.values):
                    if name not in skip_values:
                        diff_str = compute_differential_str(reference_row[name], row[name], round_decimals)
                        row[name] = str(round(row[name], round_decimals)) + diff_str
                differential_dataframe.iloc[index] = row

    else:
        raise ValueError('Results mode must be either "average" or "individual"')

    if save_csv_path is not None:
        differential_dataframe.to_csv(save_csv_path, float_format='%.' + str(round_decimals) + 'f', index=False)

    return differential_dataframe


def check_result_type(result_csv_file, individual):
    """
    Checks if the evaluation results file type is of the required format i.e. individual or average metrics
    Args:
        result_csv_file: csv file name
        individual: Boolean set to True if 'result_csv_file' is individual. Otherwise, set to False.
    Returns: True if the file 'result_csv_file' is of the required type, else False
    """
    csv_type = result_csv_file[result_csv_file.rfind('_') + 1:-4]
    if individual and csv_type == 'individual' or not individual and csv_type == 'average':
        return True
    elif individual and csv_type == 'average' or not individual and csv_type == 'individual':
        return False
    else:
        raise ValueError('File name not in required format')


def check_squash_classes(n_classes, squash_classes):
    # Check for empty lists
    for item in squash_classes:
        if not item:
            raise ValueError('Empty class value %s' % squash_classes)
    # Check all the class index are included
    allowed_classes = set(list(range(0, n_classes)))
    squashed_classes = set(list(itertools.chain.from_iterable(squash_classes)))
    if allowed_classes != squashed_classes:
        raise ValueError('Incorrect squash classes values %s' % squashed_classes.difference(allowed_classes))
    # Check len of the squashed output is less than the current output
    if len(squash_classes) >= n_classes:
        raise ValueError('Incorrect squash classes length %i maximum length is %i' % (len(squash_classes), n_classes))
    return True


def check_threshold_classes(n_classes, threshold):
    valid_keys = ['threshold_class', 'threshold_prob', 'trigger_classes']
    allowed_classes = set(list(range(0, n_classes)))
    if sorted(threshold.keys()) != sorted(valid_keys):
        raise ValueError('There are keys missing or not allowed keys in threshold dictionary')
    # Check for empty lists
    for class_idx in threshold['trigger_classes']:
        if class_idx not in allowed_classes:
            raise ValueError('Error in trigger class %i, not in the classes available %s'
                             % (class_idx, allowed_classes))
    if threshold['threshold_class'] not in allowed_classes:
        raise ValueError('Error in threshold_class class %i, not in the classes available %s'
                         % (threshold['threshold_class'], allowed_classes))

    if threshold['threshold_prob'] > 1.0 or threshold['threshold_prob'] < 0.0:
        raise ValueError('Error in threshold_prob class, it should be between 0.0 and 1.0')

    return True


def check_pass_through_classes(n_classes, pass_through_classes):
    allowed_classes = set(list(range(0, n_classes)))
    for class_idx in pass_through_classes:
        if class_idx not in allowed_classes:
            raise ValueError('Error in pass through class %i, not in the classes available %s'
                             % (class_idx, allowed_classes))
    return True


def load_yaml_file(yaml_file):
    with open(yaml_file, 'r') as f:
        loaded_yaml = yaml.safe_load(f)

    return loaded_yaml


def check_duplicate_tags_in_manifest(manifest_df, tag_id):
    """

    Args:
        manifest_df: manifest dataframe
        tag_id: tag id to compare (e.g. AIP will check for AIP codes)

    Returns: Image ids for images with duplicate tags

    """
    id_list = []
    for idx, row in manifest_df.iterrows():
        tags = [tag for tag in row.tags if tag_id in tag]
        if len(tags) > 1:
            id_list.append(row.id)
    return id_list


def check_tags_exist_in_manifest(manifest_df, tag_id):
    """

        Args:
            manifest_df: manifest dataframe
            tag_id: tag id to compare (e.g. AIP will check for AIP codes)

        Returns: Image ids for images without the tag_id introduced

        """
    id_list = []
    for idx, row in manifest_df.iterrows():
        tags = [tag for tag in row.tags if tag_id in tag]
        if len(tags) == 0:
            id_list.append(row.id)
    return id_list


def dump_yaml_file(filename, dictionary):
    mkdir('/'.join(filename.split('/')[:-1]))
    with open(filename, 'w') as yaml_file:
        yaml.dump(dictionary, yaml_file, default_flow_style=False)


def read_json(json_file):
    with open(json_file) as file:
        data = json.load(file)
    return data


def compare_visual_by_definition_results(visual_accuracy, visual_qa_accuracy):
    """
    Creates a data frame to compare the visual results with the visual-qa results.
    visual_accuracy: A list of top-k accuracies from the CNN.
    visual_qa_accuracy: A list of top-k accuracies from visual-qa.
    """
    if len(visual_accuracy) != len(visual_qa_accuracy):
        raise ValueError("The visual and visual-qa accuracy have not be computed for the same top-k")

    comparision = []
    visual_dict = {"Mode": "Visual"}
    for i in range(len(visual_accuracy)):
        key = "top_" + str(i + 1)
        visual_dict.update({key: visual_accuracy[i]})
    comparision.append(visual_dict)
    visual_qa_dict = {"Mode": "Visual_QA"}
    for i in range(len(visual_qa_accuracy)):
        key = "top_" + str(i + 1)
        visual_qa_dict.update({key: visual_qa_accuracy[i]})
    comparision.append(visual_qa_dict)

    return pd.DataFrame(comparision)


def create_differential_indx_list(differential_diagnosis, diagnosis_ids):
    """
    Create a list of indices for differential diagnosis indices
    Args:
        differential_diagnosis: A list of differentials for each test case
        diagnosis_ids: A list of indexed diagnosis ids
    Return:
        Differential diagnosis indices
    """
    differentials_indx = []
    for i in range(len(differential_diagnosis)):
        condition_differentials = []
        if len(differential_diagnosis[i]):
            for differential in differential_diagnosis[i]:
                condition_differentials.append(diagnosis_ids.index(differential))
            differentials_indx.append(condition_differentials)
        else:
            differentials_indx.append([])
    return differentials_indx


def create_multilabel_y_true(probabilities, labels, differential_diagnosis, diagnosis_ids):
    """
    Create a numpy array with multiple indices representing the true label
    Args:
        probs: A list of probabilities for all test cases
        labels: A list of the indices of ground truth conditions
        differential_diagnosis: A list of differentials represented by indices of the visual classifier
        diagnosis_ids: A list of diagnosis ids indexed by the order of the CNN's output
    Returns:
        A numpy array with true labels
    """
    differentials_indx = create_differential_indx_list(differential_diagnosis, diagnosis_ids)
    multilabel_y_true = np.zeros((probabilities.shape[0], probabilities.shape[1]))
    for i in range(len(labels)):
        multilabel_y_true[i, labels[i]] = 1
    for i in range(len(differentials_indx)):
        for indx in differentials_indx[i]:
            multilabel_y_true[i, indx] = 1
    return multilabel_y_true


def store_data_csv(dir_path, file_name, table, extra_string=''):
    """
    Converts list to CSV
    """

    with open(os.path.join(dir_path, file_name), 'w') as errors_file:
        writer = csv.writer(errors_file)
        if extra_string:
            writer.writerow(extra_string)
        for row in table:
            writer.writerow([row])


def get_n_batches(n_samples, batch_size):
    """Return the number of batches and the remainder in `samples` given a `batch_size`."""
    n_batches = n_samples // batch_size
    n_remainder = n_samples % batch_size

    return n_batches, n_remainder


def convert_results_table(individual_results_df, diagnosis_model_map_df):
    """Return a results dataframe from the individual results table derived from evaluator show_results.

    The class order of individual_results_df and diagnosis_model_map_df *must* be the same.

    Args:
        individual_results_df: A dataframe that contains results per condition.
        diagnosis_model_map_df: A diagnosis model map
    Returns: A dataframe in the required format
    """
    individual_results_df['condition'] = diagnosis_model_map_df.diagnosis_name
    individual_results_df['aip_id'] = diagnosis_model_map_df.diagnosis_id
    individual_results_df['sensitivitytopfivepercent'] = individual_results_df.apply(
        lambda x: x['sensitivity_top_5'] * 100, axis=1, result_type='expand')

    # Copy helps to manipulate the resulting dataframe.
    results_table_df = individual_results_df[['condition', 'aip_id', 'sensitivitytopfivepercent', 'n_samples']].copy()
    results_table_df['condition'] = results_table_df['condition'].str.title()
    results_table_df.sort_values(by=['condition'], inplace=True)
    results_table_df = results_table_df.rename(columns={"n_samples": "nsamples"})
    return results_table_df
