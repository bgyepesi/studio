import re
import os
import json
import pytest
import numpy as np
import pandas as pd
import studio.evaluation.keras.utils as utils


from keras_model_specs import ModelSpec


def test_create_data_generator(test_catdog_dataset_path, test_catdog_manifest_dataframe, test_catdog_mobilenet_model_spec):
    with open(test_catdog_mobilenet_model_spec) as f:
        model_spec_json = json.load(f)
        model_spec = ModelSpec(model_spec_json)

    gen, labels = utils.create_data_generator(data_dir=test_catdog_dataset_path,
                                              model_spec=model_spec,
                                              batch_size=2,
                                              dataframe=test_catdog_manifest_dataframe,
                                              custom_crop=True)
    batch_x, batch_y = gen.next()
    assert batch_x.shape == (2, 224, 224, 3)
    assert batch_y.shape == (2, 2)
    assert labels.shape == (4, 2)

    gen, labels = utils.create_data_generator(data_dir=test_catdog_dataset_path,
                                              data_augmentation={'scale_sizes': [256],
                                                                 'crop_original': 'center_crop',
                                                                 'transforms': ['horizontal_flip',
                                                                                'vertical_flip',
                                                                                'rotate_90',
                                                                                'rotate_180',
                                                                                'rotate_270']},
                                              model_spec=model_spec,
                                              batch_size=2,
                                              dataframe=test_catdog_manifest_dataframe,
                                              validate_filenames=False)

    batch_x, batch_y = gen.next()
    assert batch_x.shape == (1, 108, 224, 224, 3)
    assert len(batch_y) == 108
    assert labels.shape == (4, 2)


def test_safe_divide():
    assert np.isnan(utils.safe_divide(10.0, 0.0))
    assert utils.safe_divide(10.0, 5.0) == 2.0


def test_round_list():
    input_list = [0.6666666666, 0.3333333333]
    assert utils.round_list(input_list, decimals=2) == [0.67, 0.33]
    assert utils.round_list(input_list, decimals=4) == [0.6667, 0.3333]
    assert utils.round_list(input_list, decimals=6) == [0.666667, 0.333333]


def test_read_dictionary(test_animals_dictionary_path):
    dictionary = utils.read_dictionary(test_animals_dictionary_path)
    expected = 5
    actual = len(dictionary)
    assert actual == expected


def test_load_model(test_animals_mobilenet_path, test_animals_model_spec_path):

    # Default model_spec
    model = utils.load_model(test_animals_mobilenet_path)
    assert model

    # Custom model_spec
    model = utils.load_model(test_animals_mobilenet_path, specs_path=test_animals_model_spec_path)
    assert model


def test_load_model_ensemble(test_animals_ensemble_path):
    models, specs = utils.load_multi_model(test_animals_ensemble_path)
    assert models
    assert specs


def test_combine_probabilities():
    # Ensemble 3 models
    probabilities = np.array([[[0.4, 0.6], [0.8, 0.2]], [[0.1, 0.9], [0.2, 0.6]], [[0.4, 0.6], [0.8, 0.2]]])

    # Weighted ensemble
    ensemble_weights = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError,
                       match=re.escape('The sum of the weights provided (%f) do not aggregate to 1.0'
                                       % (np.sum(ensemble_weights)))):
        utils.combine_probabilities(probabilities, 'arithmetic', ensemble_weights=ensemble_weights)

    ensemble_weights = [0.5, 0.5]
    with pytest.raises(ValueError,
                       match='Length of weights %d do not coincide with the number of models %d'
                             % (len(ensemble_weights), probabilities.shape[0])):
        utils.combine_probabilities(probabilities, 'arithmetic', ensemble_weights=ensemble_weights)

    ensemble_weights = [0.5, 0.5, 0.0]
    combined_probabilities = utils.combine_probabilities(probabilities, 'arithmetic', ensemble_weights=ensemble_weights)
    combined_probabilities_expected = [[0.25, 0.75], [0.5, 0.4]]
    np.testing.assert_array_equal(np.round(combined_probabilities, decimals=2), combined_probabilities_expected)

    # Maximum
    combined_probabilities = utils.combine_probabilities(probabilities, 'maximum')
    assert len(combined_probabilities.shape) == 2
    combined_probabilities_expected = [[0.4, 0.9], [0.8, 0.6]]
    np.testing.assert_array_equal(np.round(combined_probabilities, decimals=2), combined_probabilities_expected)

    # Arithmetic
    combined_probabilities = utils.combine_probabilities(probabilities, 'arithmetic')
    assert len(combined_probabilities.shape) == 2
    combined_probabilities_expected = [[0.3, 0.7], [0.6, 0.33]]
    np.testing.assert_array_equal(np.round(combined_probabilities, decimals=2), combined_probabilities_expected)

    # Geometric
    combined_probabilities = utils.combine_probabilities(probabilities, 'geometric')
    assert len(combined_probabilities.shape) == 2
    combined_probabilities_expected = [[0.25, 0.69], [0.5, 0.29]]
    np.testing.assert_array_equal(np.round(combined_probabilities, decimals=2), combined_probabilities_expected)

    # Harmonic
    combined_probabilities = utils.combine_probabilities(probabilities, 'harmonic')
    assert len(combined_probabilities.shape) == 2
    combined_probabilities_expected = [[0.2, 0.68], [0.4, 0.26]]
    np.testing.assert_array_equal(np.round(combined_probabilities, decimals=2), combined_probabilities_expected)

    # One model, ndim = 3
    probabilities = np.array([[0.4, 0.6], [0.8, 0.2]])
    probabilities_exp = np.array(np.expand_dims(probabilities, axis=0))
    assert probabilities_exp.shape == (1, 2, 2)

    combined_probabilities = utils.combine_probabilities(probabilities_exp, 'maximum')
    assert combined_probabilities.shape == (2, 2)
    np.testing.assert_array_equal(combined_probabilities, probabilities)

    # One model, ndim=2
    probabilities = np.array([[0.4, 0.6], [0.8, 0.2]])
    assert probabilities.shape == (2, 2)
    combined_probabilities = utils.combine_probabilities(probabilities)
    assert combined_probabilities.shape == (2, 2)
    np.testing.assert_array_equal(combined_probabilities, probabilities)


def test_get_valid_images():
    image_list_test = ['a.jpg', 'b.sdji', 'u.png', 'wqd.jpeg']
    filtered_list = utils.get_valid_images(image_list_test)

    assert filtered_list == ['a.jpg', 'u.png', 'wqd.jpeg']


def test_load_preprocess_image(test_image_path, model_spec_mobilenet):
    image = utils.load_preprocess_image(test_image_path, model_spec_mobilenet)
    assert image.shape == (1, 224, 224, 3)


def test_load_preprocess_image_list(test_image_paths_list, model_spec_mobilenet):
    images = utils.load_preprocess_image_list(test_image_paths_list, model_spec_mobilenet)
    assert np.array(images).shape == (2, 224, 224, 3)


def test_default_concepts(test_catdog_dataset_path):
    concepts_by_default = utils.get_default_concepts(test_catdog_dataset_path)
    assert concepts_by_default == [{'label': 'cat', 'id': 'cat'},
                                   {'label': 'dog', 'id': 'dog'}]


def test_get_dictionary_concepts(test_animals_dictionary_path):
    dictionary_concepts = utils.get_dictionary_concepts(test_animals_dictionary_path)
    assert dictionary_concepts == [{'label': '00000_cat', 'id': 0},
                                   {'label': '00001_dog', 'id': 1},
                                   {'label': '00002_goose', 'id': 2},
                                   {'label': '00003_turtle', 'id': 3},
                                   {'label': '00004_elephant', 'id': 4}]


def test_create_training_json(test_catdog_dataset_path, test_dir_tests):
    dict_path = os.path.join(test_dir_tests, 'dict.json')
    utils.create_training_json(test_catdog_dataset_path, dict_path)
    actual = os.path.isfile(dict_path)
    expected = True
    assert actual == expected


def test_compare_concept_dictionaries():
    concept_lst = ['dog', 'elephant']
    concept_dict = [{'group': 'dog'}, {'group': 'cat'}, {'group': 'elephant'}]
    with pytest.raises(ValueError,
                       match=re.escape("('The following concepts are not present in either the concept dictionary or "
                                       "among the test classes:', ['cat'])")):
        utils.compare_group_test_concepts(concept_lst, concept_dict)


def test_check_input_samples(metrics_top_k_multi_class):
    _, y_true, y_probs = metrics_top_k_multi_class
    assert utils.check_input_samples(y_probs, y_true)

    y_true = np.asarray([0, 1, 2, 2, 1])  # 5 samples, 3 classes.
    y_probs = np.asarray([[1, 0, 0], [0.2, 0.2, 0.6], [0.8, 0.2, 0], [0.35, 0.25, 0.4]])  # 4 samples, 3 classes.
    with pytest.raises(ValueError,
                       match=re.escape('The number predicted samples (4) is different from '
                                       'the ground truth samples (5)')):
        utils.check_input_samples(y_probs, y_true)


def test_check_concept_unique():
    concept_dict = [{'class_name': 'cat'}, {'class_name': 'dog'}, {'class_name': 'cat'}]
    with pytest.raises(ValueError, match="('Concept has been repeated:', 'cat')"):
        utils.check_concept_unique(concept_dict)


def test_get_class_dictionaries_items(test_catdog_dataset_path):
    concepts_by_default = utils.get_default_concepts(test_catdog_dataset_path)
    label_output = utils.get_concept_items(concepts_by_default, 'label')
    id_output = utils.get_concept_items(concepts_by_default, 'id')
    assert label_output == id_output == ['cat', 'dog']


def test_results_to_dataframe():
    results = {'individual':
               [{
                   'id': 'Class_0',
                   'label': 'Class_0',
                   'metrics': {'TP': 2, 'precision': 1.0, 'AUROC': 0.8333333, 'sensitivity': 1.0,
                                        'FN': 0, 'FDR': 0.0, 'f1_score': 1.0, 'FP': 0}},
                {
                    'id': 'Class_1',
                    'label': 'Class_1',
                    'metrics': {'TP': 2, 'precision': 1.0, 'AUROC': 0.8333333,
                                'sensitivity': 1.0, 'FN': 0, 'FDR': 0.0,
                                'f1_score': 1.0, 'FP': 0}}],
               'average': {'precision': [1.0], 'confusion_matrix': np.array([[2, 0], [0, 2]]), 'sensitivity': [1.0],
                           'auroc': [0.8333333], 'f1_score': [1.0], 'accuracy': [1.0],
                           'specificity': [1.0], 'fdr': [0.0]}}

    # Assert error when incorrect mode
    with pytest.raises(ValueError, match='Results mode must be either "average" or "individual"'):
        utils.results_to_dataframe(results, mode='asdf')

    average_df = utils.results_to_dataframe(results)
    assert average_df['model_id'][0] == 'default_model'
    assert average_df['accuracy'][0] == average_df['precision'][0] == average_df['f1_score'][0] == 1.0

    individual_df = utils.results_to_dataframe(results, mode='individual')
    assert individual_df['class'][0] == 'Class_0'
    assert individual_df['class'][1] == 'Class_1'
    assert individual_df['sensitivity'][0] == individual_df['sensitivity'][1] == 1.0
    assert individual_df['precision'][0] == individual_df['precision'][1] == 1.0
    assert individual_df['f1_score'][0] == individual_df['f1_score'][1] == 1.0
    assert individual_df['TP'][0] == individual_df['TP'][1] == 2
    assert individual_df['FP'][0] == individual_df['FP'][1] == individual_df['FN'][1] == individual_df['FN'][1] == 0
    assert individual_df['AUROC'][0] == individual_df['AUROC'][1] == 0.833


def test_ensemble_models(test_image_path, test_catdog_ensemble_path):
    models, model_specs = utils.load_multi_model(test_catdog_ensemble_path)

    with pytest.raises(ValueError, match='Incorrect combination mode selected, we only allow for `average` '
                                         'or `maximum`'):
        utils.ensemble_models(models, input_shape=(224, 224, 3), combination_mode='asdf')

    with pytest.raises(ValueError, match=re.escape('Incorrect input shape, it should have 3 dimensions (H, W, C)')):
        utils.ensemble_models(models, input_shape=(224, 3), combination_mode='asdf')

    ensemble = utils.ensemble_models(models, input_shape=(224, 224, 3), combination_mode='average')
    image = utils.load_preprocess_image(test_image_path, model_specs[0])

    # forward pass
    preds = ensemble.predict(image)
    # 1 sample
    assert preds.shape[0] == 1
    # 2 predictions
    assert preds.shape[1] == 2


def test_load_csv_to_dataframe(test_average_results_csv_paths):
    # No error
    assert len(utils.load_csv_to_dataframe(test_average_results_csv_paths)) == 3

    # Format error
    with pytest.raises(ValueError, match='Incorrect format for `csv_paths`, a list of strings or a single '
                                         'string are expected'):
        utils.load_csv_to_dataframe(1)


def test_results_differential(test_average_results_csv_paths, test_individual_results_csv_paths):
    with pytest.raises(ValueError, match='The number of dataframes should be higher than 1'):
        utils.results_differential(['asd'])

    dataframes = utils.load_csv_to_dataframe(test_average_results_csv_paths)
    df = utils.results_differential(dataframes, mode='average')
    assert len(df) == 3
    assert df['accuracy'][1] == '0.2 (-0.6)'
    assert df['accuracy'][2] == '0.9 (+0.1)'
    assert df['weighted_precision'][1] == '0.2 (-0.4)'
    assert df['precision'][2] == '0.3 (-0.355)'
    assert df['sensitivity'][1] == '0.2 (0.0)'
    assert df['f1_score'][2] == '0.3 (-0.45)'
    assert df['number_of_samples'][1] == 2000
    assert df['number_of_samples'][2] == 2000
    assert df['number_of_classes'][1] == 2
    assert df['number_of_classes'][2] == 2

    dataframes = utils.load_csv_to_dataframe(test_individual_results_csv_paths)
    df = utils.results_differential(dataframes, mode='individual')

    assert len(df) == 6

    sensitivity_values_expected = [0.9, '0.2 (-0.7)', '0.15 (-0.75)', 0.1, '0.1 (0.0)', '0.25 (+0.15)']
    for i, val in enumerate(df['sensitivity']):
        assert val == sensitivity_values_expected[i]

    precision_values_expected = [0.55, '0.4 (-0.15)', '0.35 (-0.2)', 0.2, '0.3 (+0.1)', '0.65 (+0.45)']
    for i, val in enumerate(df['precision']):
        assert val == precision_values_expected[i]

    percentage_samples_values_expected = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    for i, val in enumerate(df['% of samples']):
        assert val == percentage_samples_values_expected[i]


def test_compute_differential_str():
    assert utils.compute_differential_str(0.90, 0.65, 4) == ' (-0.25)'
    assert utils.compute_differential_str(0.65, 0.90, 4) == ' (+0.25)'


def test_check_squash_classes(get_first_gate_config):
    config = get_first_gate_config
    n_outputs = 4
    assert utils.check_squash_classes(n_outputs, config['squash_classes'])

    config['squash_classes'] = [[0], [1], [2], [3], [5]]
    with pytest.raises(ValueError, match=re.escape('Incorrect squash classes values %s' % {5})):
        utils.check_squash_classes(n_outputs, config['squash_classes'])

    n_outputs = 4
    config['squash_classes'] = [[0], [1], [2], [3], [3]]
    with pytest.raises(ValueError, match=re.escape('Incorrect squash classes length %i maximum length is %i'
                                                   % (5, n_outputs))):
        utils.check_squash_classes(n_outputs, config['squash_classes'])

    n_outputs = 4
    config['squash_classes'] = [[0], [], [2], [3]]
    with pytest.raises(ValueError, match=re.escape('Empty class value %s' % config['squash_classes'])):
        utils.check_squash_classes(n_outputs, config['squash_classes'])


def test_check_threshold_classes(get_first_gate_config):
    config = get_first_gate_config
    n_outputs = 4
    assert utils.check_threshold_classes(n_outputs, config['threshold'])
    allowed_classes = set(list(range(0, n_outputs)))

    config['threshold'] = {
        'threshold_class': 3,
        'threshold_prob': 0.1,
    }

    with pytest.raises(ValueError, match='There are keys missing or not allowed keys in threshold dictionary'):
        utils.check_threshold_classes(n_outputs, config['threshold'])

    config['threshold'] = {
        'threshold_class': 2,
        'threshold_prob': 0.1,
        'trigger_classes': [5]
    }

    with pytest.raises(ValueError, match='Error in trigger class %i, not in the classes available %s'
                                         % (5, allowed_classes)):
        utils.check_threshold_classes(n_outputs, config['threshold'])

    config['threshold'] = {
        'threshold_class': 5,
        'threshold_prob': 0.1,
        'trigger_classes': [1]
    }

    with pytest.raises(ValueError, match='Error in threshold_class class %i, not in the classes available %s'
                                         % (config['threshold']['threshold_class'], allowed_classes)):
        utils.check_threshold_classes(n_outputs, config['threshold'])

    config['threshold'] = {
        'threshold_class': 2,
        'threshold_prob': 1.5,
        'trigger_classes': [1]
    }

    with pytest.raises(ValueError, match='Error in threshold_prob class, it should be between 0.0 and 1.0'):
        utils.check_threshold_classes(n_outputs, config['threshold'])


def test_check_pass_through_classes(get_first_gate_config):
    n_outputs = 4
    config = get_first_gate_config
    assert utils.check_pass_through_classes(n_outputs, config['pass_through'])

    allowed_classes = set(list(range(0, n_outputs)))
    config['pass_through'] = [0, 1, 5]
    with pytest.raises(ValueError, match='Error in pass through class %i, not in the classes available %s' %
                                         (5, allowed_classes)):
        utils.check_pass_through_classes(n_outputs, config['pass_through'])


def test_load_file_yaml(first_gate_config_yaml):
    config = utils.load_yaml_file(first_gate_config_yaml)
    expected_config = {
        'id': 'first_gate',
        'model_map_json': os.path.abspath(os.path.join('tests', 'evaluation', 'keras', 'files', 'config_files',
                                                       'first_gate.json')),
        'batch_size': 1,
        'squash_classes': [[0, 1], [2], [3]],
        'threshold': {
            'threshold_class': 2,
            'threshold_prob': 0.1,
            'trigger_classes': [1]},
        'pass_through': [2],
    }
    assert config == expected_config


def test_compare_visual_by_definition_results():
    visual_accuracy = [0.3, 0.4, 0.5]
    visual__qa_accuracy = [0.3, 0.4, 0.5, 0.6]
    with pytest.raises(ValueError, match='The visual and visual-qa accuracy have not be computed for the same top-k'):
        utils.compare_visual_by_definition_results(visual_accuracy, visual__qa_accuracy)


def test_create_differential_indx_list():
    ddx_lst = [["b", "c", "d"], ["d", "a"]]
    diagnosis_ids = ["a", "b", "c", "d"]
    expected = [[1, 2, 3], [3, 0]]
    actual = utils.create_differential_indx_list(ddx_lst, diagnosis_ids)
    assert actual == expected


def test_create_multilabel_y_true():
    probs = np.array([[0.4, 0.3, 0.2, 0.1], [0.5, 0, 0, 0.5]])
    labels = [0, 3]
    ddx_lst = [["b", "c", "d"], ["d", "a"]]
    diagnosis_ids = ["a", "b", "c", "d"]
    expected = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    actual = utils.create_multilabel_y_true(probs, labels, ddx_lst, diagnosis_ids)
    np.testing.assert_array_equal(actual, expected)


def test_convert_results_table():
    model_map_dict = {"diagnosis_name": ['moha', 'albert', 'acne vul'], "diagnosis_id": ["AIP:1", "AIP:2", "AIP:3"]}
    diagnosis_model_map_df = pd.DataFrame(data=model_map_dict)
    # The only columns needed are 'sensitivity_top_5' and 'n_samples'
    individual_results_dict = {
        "dummy_values": [1, 2, 3],
        "sensitivity_top_5": [0.45, 0.74, 0.99],
        "n_samples": [5, 10, 9]
    }
    individual_results_df = pd.DataFrame(data=individual_results_dict)
    results_table_df = utils.convert_results_table(individual_results_df, diagnosis_model_map_df)
    expected_columns = ['condition', 'aip_id', 'sensitivitytopfivepercent', 'nsamples']
    actual_columns = list(results_table_df.columns)

    assert actual_columns == expected_columns
    expected_df_shape = (len(individual_results_dict['dummy_values']), len(expected_columns))
    actual_df_shape = results_table_df.shape

    assert actual_df_shape == expected_df_shape
    print(results_table_df)
    assert results_table_df.iloc[0]['sensitivitytopfivepercent'] == 99
