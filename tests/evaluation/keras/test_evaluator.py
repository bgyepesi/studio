import os
import pytest
import numpy as np

from studio.evaluation.keras import utils


def test_plot_confusion_matrix(evaluator_catdog_mobilenet, test_catdog_dataset_path):
    evaluator_catdog_mobilenet.evaluate(test_catdog_dataset_path, confusion_matrix=True)


def test_set_concepts(evaluator_catdog_mobilenet):
    with pytest.raises(ValueError,
                       match='Incorrect format for concepts list. It must contain the fields `id` and `label`'):
        evaluator_catdog_mobilenet.set_concepts([{'id': 'abcd', 'label': 'asd'}, {'a': 'b', 'b': 'c'}])

    evaluator_catdog_mobilenet.set_concepts([{'id': '1', 'label': '1'}, {'id': '2', 'label': '2'}])


def test_set_combination_mode(evaluator_catdog_mobilenet):
    with pytest.raises(ValueError,
                       match='Error: invalid option for `combination_mode` asdf'):
        evaluator_catdog_mobilenet.set_combination_mode('asdf')

    evaluator_catdog_mobilenet.set_combination_mode('maximum')


def check_evaluate_on_catdog_dataset(evaluator, test_catdog_dataset_path):
    evaluator.evaluate(test_catdog_dataset_path)
    probabilities, labels = evaluator.probabilities, evaluator.labels

    # n_samples x n_classes
    assert len(probabilities.shape) == 2

    # n_samples x n_classes
    assert len(labels.shape) == 1

    # n_classes = 2
    assert evaluator.num_classes == 2

    # class abbreviations
    assert evaluator.concept_labels == ['cat', 'dog']


def check_evaluate_on_catdog_dataset_dataframe(evaluator, test_catdog_dataset_path, test_catdog_manifest_path):
    evaluator.evaluate(test_catdog_dataset_path, dataframe_path=test_catdog_manifest_path)
    probabilities, labels = evaluator.probabilities, evaluator.labels

    # n_samples x n_classes
    assert len(probabilities.shape) == 2

    # n_samples x n_classes
    assert len(labels.shape) == 1

    # n_classes = 2
    assert evaluator.num_classes == 2

    # class abbreviations
    assert evaluator.concept_labels == ['class_0', 'class_1']


def test_evaluate_on_catdog_dataset_dataframe_with_tag(cnn_evaluator_tag_catdog_mobilenet,
                                                       test_catdog_dataset_tag_path,
                                                       test_catdog_manifest_tag_path):
    cnn_evaluator_tag_catdog_mobilenet.evaluate(test_catdog_manifest_tag_path, data_dir=test_catdog_dataset_tag_path,
                                                tag="case-id")
    # Get probabilities and labels
    probabilities, labels = cnn_evaluator_tag_catdog_mobilenet.probabilities, cnn_evaluator_tag_catdog_mobilenet.labels
    # Get tags the the dataframe containing the cases with different labels
    tags, different_label_cases_df = cnn_evaluator_tag_catdog_mobilenet.tags, cnn_evaluator_tag_catdog_mobilenet.different_label_cases_df

    # check the number of total cases
    assert len(tags.tolist()) == 9

    # check the number of unique cases
    assert len(set(tags.tolist())) == 6

    # Check that the two images corresponding to the case-id tag `dog-case`
    # are in the dataframe containing the cases with different labels.
    assert set(different_label_cases_df['case-id'].tolist()) == {"dog-case3"}

    # Check that the number of probabilities' vectors is equal
    # to the number of unique cases minus the cases with different labels.
    assert probabilities.shape[0] == len(set(tags.tolist())) - len(set(different_label_cases_df['case-id'].tolist()))
    assert len(labels) == len(set(tags.tolist())) - len(set(different_label_cases_df['case-id'].tolist()))


def check_predict_on_cat_folder(evaluator, test_cat_folder):
    probabilities = evaluator.predict(test_cat_folder)

    # n_samples x n_classes
    assert len(probabilities.shape) == 2

    # 2 images in the folder
    assert len(evaluator.image_paths) == 2


def check_predict_on_image_paths(evaluator, image_paths_list):
    probabilities = evaluator.predict(data_dir=image_paths_list)

    # n_samples x n_classes
    assert len(probabilities.shape) == 2

    # 2 images in the folder
    assert len(evaluator.image_paths) == 2


def check_predict_single_image(evaluator, test_image_path):
    probabilities = evaluator.predict(test_image_path)

    # n_samples x n_classes
    assert len(probabilities.shape) == 2

    # 1 image predicted
    assert len(evaluator.image_paths) == 1


def test_check_compute_probabilities_generator_data_augmentation(evaluator_catdog_mobilenet,
                                                                 test_catdog_dataset_path):
    evaluator_catdog_mobilenet.evaluate(test_catdog_dataset_path,
                                        data_augmentation={
                                            'scale_sizes': [256],
                                            'transforms': ['horizontal_flip'],
                                            'crop_original': 'center_crop'}
                                        )
    probabilities = evaluator_catdog_mobilenet.probabilities
    assert probabilities.shape == (4, 2)

    for p in probabilities:
        np.testing.assert_almost_equal(sum(p), 1.0, decimal=5)


def test_check_compute_probabilities_generator_dataframe_data_augmentation(evaluator_catdog_mobilenet,
                                                                           test_catdog_dataset_path,
                                                                           test_catdog_manifest_path):
    evaluator_catdog_mobilenet.evaluate(test_catdog_dataset_path,
                                        dataframe_path=test_catdog_manifest_path,
                                        custom_crop=True,
                                        data_augmentation={
                                            'scale_sizes': [256],
                                            'transforms': ['horizontal_flip'],
                                            'crop_original': 'center_crop'}
                                        )
    probabilities = evaluator_catdog_mobilenet.probabilities
    assert probabilities.shape == (4, 2)

    for p in probabilities:
        np.testing.assert_almost_equal(sum(p), 1.0, decimal=5)


def test_check_evaluate_class_inference_mobilenet(evaluator_animals_mobilenet_class_inference,
                                                  test_animals_dataset_path):
    evaluator_animals_mobilenet_class_inference.evaluate(test_animals_dataset_path)
    probabilities = evaluator_animals_mobilenet_class_inference.probabilities
    assert probabilities.shape == (15, 3)

    for p in probabilities:
        np.testing.assert_almost_equal(sum(p), 1.0, decimal=5)


def test_check_compute_inference_probabilities_mobilenet(evaluator_animals_mobilenet_class_inference,
                                                         test_animals_dataset_path,
                                                         test_animals_dictionary_path):
    evaluator_animals_mobilenet_class_inference.data_dir = test_animals_dataset_path
    evaluator_animals_mobilenet_class_inference.concepts = utils.get_dictionary_concepts(test_animals_dictionary_path)
    group_concepts = utils.get_default_concepts(evaluator_animals_mobilenet_class_inference.data_dir)
    evaluator_animals_mobilenet_class_inference.concept_labels = \
        utils.get_concept_items(concepts=group_concepts, key='label')

    probabilities, labels = evaluator_animals_mobilenet_class_inference._compute_probabilities_generator(
        evaluator_animals_mobilenet_class_inference.data_dir)
    probabilities = utils.combine_probabilities(probabilities, 'arithmetic')
    inference_probabilities = evaluator_animals_mobilenet_class_inference._compute_inference_probabilities(
        probabilities)

    assert inference_probabilities.shape == (15, 3)

    for p in inference_probabilities:
        np.testing.assert_almost_equal(sum(p), 1.0, decimal=5)


def test_check_evaluate_class_inference_ensemble(evaluator_animals_ensemble_class_inference,
                                                 test_animals_dataset_path):
    evaluator = evaluator_animals_ensemble_class_inference
    evaluator.evaluate(test_animals_dataset_path)
    probabilities = evaluator.probabilities
    assert probabilities.shape == (15, 3)

    for p in probabilities:
        np.testing.assert_almost_equal(sum(p), 1.0, decimal=5)


def test_get_image_paths_by_prediction(evaluator_catdog_mobilenet, test_catdog_dataset_path, test_cat_folder,
                                       test_dog_folder):
    evaluator_catdog_mobilenet.evaluate(test_catdog_dataset_path)
    probabilities = evaluator_catdog_mobilenet.probabilities
    labels = evaluator_catdog_mobilenet.labels
    image_dictionary = evaluator_catdog_mobilenet.get_image_paths_by_prediction(probabilities, labels)

    assert image_dictionary['cat_cat']['image_paths'] == [os.path.join(test_cat_folder, 'cat-4.jpg')]
    assert len(image_dictionary['cat_cat']['probs']) == 1
    assert image_dictionary['cat_dog']['image_paths'] == [os.path.join(test_cat_folder, 'cat-1.jpg')]
    assert len(image_dictionary['cat_dog']['probs']) == 1
    assert image_dictionary['dog_cat']['image_paths'] == []
    assert len(image_dictionary['dog_cat']['probs']) == 0
    assert image_dictionary['dog_dog']['image_paths'] == [os.path.join(test_dog_folder, 'dog-2.jpg'),
                                                          os.path.join(test_dog_folder, 'dog-4.jpg')]
    assert len(image_dictionary['dog_dog']['probs']) == 2


def test_evaluator_single_mobilenet_v1_on_catdog_dataset(evaluator_catdog_mobilenet, test_catdog_dataset_path,
                                                         test_cat_folder, test_image_path, test_image_paths_list):
    check_evaluate_on_catdog_dataset(evaluator_catdog_mobilenet, test_catdog_dataset_path)

    check_predict_on_cat_folder(evaluator_catdog_mobilenet, test_cat_folder)

    check_predict_single_image(evaluator_catdog_mobilenet, test_image_path)

    check_predict_on_image_paths(evaluator_catdog_mobilenet, test_image_paths_list)


def test_evaluator_single_mobilenet_v1_on_catdog_dataframe_dataset(evaluator_catdog_mobilenet,
                                                                   test_catdog_dataset_path,
                                                                   test_catdog_manifest_path):
    check_evaluate_on_catdog_dataset_dataframe(evaluator_catdog_mobilenet, test_catdog_dataset_path,
                                               test_catdog_manifest_path)


def test_evaluator_catdog_ensemble_on_catdog_dataset(evaluator_catdog_ensemble, test_catdog_dataset_path,
                                                     test_cat_folder, test_image_path, test_image_paths_list):
    check_evaluate_on_catdog_dataset(evaluator_catdog_ensemble, test_catdog_dataset_path)

    check_predict_on_cat_folder(evaluator_catdog_ensemble, test_cat_folder)

    check_predict_single_image(evaluator_catdog_ensemble, test_image_path)

    check_predict_on_image_paths(evaluator_catdog_ensemble, test_image_paths_list)


def test_show_results(fake_evaluator_catdog_mobilenet, test_catdog_dataset_path):
    # Assert error without results
    with pytest.raises(ValueError, match='results parameter is None, please run an evaluation first'):
        fake_evaluator_catdog_mobilenet.show_results('average')

    fake_evaluator_catdog_mobilenet.evaluate(test_catdog_dataset_path)

    average_df = fake_evaluator_catdog_mobilenet.show_results(mode='average')
    assert average_df['model_id'][0] == 'catdog-mobilenet.hdf5'
    assert average_df['accuracy'][0] == 0.75
    assert average_df['sensitivity'][0] == 0.75
    assert average_df['weighted_precision'][0] == 0.833
    assert average_df['precision'][0] == 0.833
    assert average_df['weighted_f1_score'][0] == 0.733

    individual_df = fake_evaluator_catdog_mobilenet.show_results(mode='individual')
    assert individual_df['class'][0] == 'cat'
    assert individual_df['class'][1] == 'dog'
    assert individual_df['sensitivity'][0] == 1.0
    assert individual_df['sensitivity'][1] == 0.5
    np.testing.assert_almost_equal(individual_df['precision'][0], 0.6669999)
    assert individual_df['precision'][1] == 1.0
    assert individual_df['f1_score'][0] == 0.8
    assert individual_df['f1_score'][1] == 0.667
    assert individual_df['TP'][0] == 2
    assert individual_df['TP'][1] == individual_df['FP'][0] == individual_df['FN'][1] == 1
    assert individual_df['FP'][1] == individual_df['FN'][0] == 0
    assert individual_df['AUROC'][0] == individual_df['AUROC'][1] == 0.75


def test_compute_confidence_prediction_distribution(fake_evaluator_catdog_mobilenet, test_catdog_dataset_path):
    with pytest.raises(ValueError, match='probabilities value is None, please run an evaluation first'):
        fake_evaluator_catdog_mobilenet.compute_confidence_prediction_distribution()

    fake_evaluator_catdog_mobilenet.evaluate(test_catdog_dataset_path)

    output = fake_evaluator_catdog_mobilenet.compute_confidence_prediction_distribution()

    np.testing.assert_array_almost_equal(output, np.array([0.82156974, 0.1784302], dtype=np.float32))


def test_compute_uncertainty_distribution(fake_evaluator_catdog_mobilenet, test_catdog_dataset_path):
    with pytest.raises(ValueError, match='probabilities value is None, please run an evaluation first'):
        fake_evaluator_catdog_mobilenet.compute_uncertainty_distribution()

    fake_evaluator_catdog_mobilenet.evaluate(test_catdog_dataset_path)

    output = fake_evaluator_catdog_mobilenet.compute_uncertainty_distribution()

    np.testing.assert_array_almost_equal(output,
                                         np.array([0.8283282, 0.13131963, 0.36905038, 0.9456398], dtype=np.float32))


def test_plot_top_k_accuracy(fake_evaluator_catdog_mobilenet):
    with pytest.raises(ValueError):
        fake_evaluator_catdog_mobilenet.plot_top_k_accuracy()


def test_plot_top_k_sensitivity_by_concept(fake_evaluator_catdog_mobilenet):
    with pytest.raises(ValueError):
        fake_evaluator_catdog_mobilenet.plot_top_k_sensitivity_by_concept()


def test_save_results(fake_evaluator_catdog_mobilenet):
    # Assert error without results
    with pytest.raises(ValueError):
        fake_evaluator_catdog_mobilenet.save_results('average', csv_path='')


def test_plot_probability_histogram(fake_evaluator_catdog_mobilenet, test_catdog_dataset_path):
    with pytest.raises(ValueError, match='There are not computed probabilities. Please run an evaluation first.'):
        fake_evaluator_catdog_mobilenet.plot_probability_histogram()

    fake_evaluator_catdog_mobilenet.evaluate(test_catdog_dataset_path)


def test_plot_most_confident(fake_evaluator_catdog_mobilenet, test_catdog_dataset_path):
    with pytest.raises(ValueError, match='There are not computed probabilities. Please run an evaluation first.'):
        fake_evaluator_catdog_mobilenet.plot_most_confident()

    fake_evaluator_catdog_mobilenet.evaluate(test_catdog_dataset_path)

    with pytest.raises(ValueError, match='Incorrect mode. Supported modes are "errors" and "correct"'):
        fake_evaluator_catdog_mobilenet.plot_most_confident(mode='x')


def test_plot_confidence_interval(fake_evaluator_catdog_mobilenet, test_catdog_dataset_path):
    with pytest.raises(ValueError, match='There are not computed probabilities. Please run an evaluation first.'):
        fake_evaluator_catdog_mobilenet.plot_confidence_interval()

    fake_evaluator_catdog_mobilenet.evaluate(test_catdog_dataset_path)

    with pytest.raises(ValueError, match='Incorrect mode. Modes available are "accuracy" or "error".'):
        fake_evaluator_catdog_mobilenet.plot_confidence_interval(mode='x')


def test_plot_sensitivity_per_samples(fake_evaluator_catdog_mobilenet):
    with pytest.raises(ValueError):
        fake_evaluator_catdog_mobilenet.plot_sensitivity_per_samples()


def test_get_sensitivity_per_samples(fake_evaluator_catdog_mobilenet, test_catdog_dataset_path):
    with pytest.raises(ValueError):
        fake_evaluator_catdog_mobilenet.get_sensitivity_per_samples()

    fake_evaluator_catdog_mobilenet.evaluate(test_catdog_dataset_path)
    results_classes = fake_evaluator_catdog_mobilenet.get_sensitivity_per_samples()

    assert results_classes['sensitivity'][0] == 1.0
    assert results_classes['sensitivity'][1] == 0.5
    assert results_classes['class'][0] == 'cat'
    assert results_classes['class'][1] == 'dog'
    assert results_classes['% samples'][0] == 50.0
    assert results_classes['% samples'][1] == 50.0


def test_ensemble_models(evaluator_catdog_ensemble, test_cat_folder):
    ensemble = evaluator_catdog_ensemble.ensemble_models(input_shape=(224, 224, 3), combination_mode='average')
    model_spec = evaluator_catdog_ensemble.model_specs[0]
    image = utils.load_preprocess_image(os.path.join(test_cat_folder, 'cat-1.jpg'), model_spec)

    # forward pass
    preds = ensemble.predict(image)
    # 1 sample
    assert preds.shape[0] == 1
    # 2 predictions
    assert preds.shape[1] == 2


def test_get_get_keys_confusion_matrix_errors(fake_evaluator_catdog_mobilenet, test_catdog_dataset_path):
    fake_evaluator_catdog_mobilenet.evaluate(test_catdog_dataset_path)
    sorted_names, sorted_counts = fake_evaluator_catdog_mobilenet.get_keys_confusion_matrix_errors()
    np.testing.assert_equal(np.array(['dog_cat', 'cat_dog']), sorted_names)
    np.testing.assert_equal(np.array([1, 0]), sorted_counts)


def test_get_errors_confusion_matrix_df(fake_evaluator_catdog_mobilenet, test_catdog_dataset_path):
    fake_evaluator_catdog_mobilenet.evaluate(test_catdog_dataset_path)
    df = fake_evaluator_catdog_mobilenet.get_errors_confusion_matrix_df()
    assert df['matrix_square'][0] == 'dog_cat'
    assert df['count'][0] == 1


def test_sequential_cnn_evaluator_evaluate(sequential_cnn_evaluator, test_image_paths_list_sequential,
                                           test_labels_sequential):
    sequential_cnn_evaluator.evaluate(test_image_paths_list_sequential, test_labels_sequential)

    probabilities = sequential_cnn_evaluator.probabilities
    labels = sequential_cnn_evaluator.labels

    np.testing.assert_almost_equal(np.array([[0., 0., 0.738949, 0.26105103],
                                             [0.55000001, 0.1, 0., 0.],
                                             [0., 0., 0.92916, 0.07083996],
                                             [0.85000002, 0.1, 0., 0.]]), probabilities)

    np.testing.assert_equal(labels, np.array([2, 2, 3, 3]))

    average_df = sequential_cnn_evaluator.show_results(mode='average')
    assert average_df['accuracy'][0] == 0.25
    assert average_df['sensitivity'][0] == 0.25
    assert average_df['weighted_precision'][0] == 0.25
    assert average_df['precision'][0] == 0.25
    assert average_df['weighted_f1_score'][0] == 0.25

    individual_df = sequential_cnn_evaluator.show_results(mode='individual')
    assert individual_df['class'][0] == 'a_b'
    assert individual_df['class'][1] == 'c'
    assert individual_df['class'][2] == 'cat'
    assert individual_df['class'][3] == 'dog'
    assert individual_df['sensitivity'][2] == 0.5
    assert individual_df['sensitivity'][3] == 0.0
    assert individual_df['precision'][0] == 0.0
    assert individual_df['precision'][2] == 0.5
    assert individual_df['f1_score'][2] == 0.5
    assert individual_df['TP'][0] == 0
    assert individual_df['FP'][0] == 2
    assert individual_df['FN'][0] == 0
    assert individual_df['TP'][2] == individual_df['FP'][2] == individual_df['FN'][2] == 1
    assert individual_df['FP'][1] == individual_df['FN'][0] == 0
    assert individual_df['% samples'][0] == individual_df['% samples'][1] == 0.0
    assert individual_df['% samples'][2] == individual_df['% samples'][3] == 50.0
    assert individual_df['n_samples'][2] == individual_df['n_samples'][3] == 2


def test_sequential_cnn_evaluator_predict(sequential_cnn_evaluator, test_image_paths_list_sequential):
    probabilities = sequential_cnn_evaluator.predict(test_image_paths_list_sequential)

    np.testing.assert_almost_equal(np.array([[0., 0., 0.738949, 0.26105103],
                                             [0.55000001, 0.1, 0., 0.],
                                             [0., 0., 0.92916, 0.07083996],
                                             [0.85000002, 0.1, 0., 0.]]), probabilities)


def test_qa_evaluate(qa_evaluator):
    np.random.seed(42)
    assert len(qa_evaluator.filtered_qa_data) == 3
    probs = []
    for i in range(len(qa_evaluator.filtered_qa_data)):
        probs.append(np.random.dirichlet(np.ones(len(
            qa_evaluator.visual_qa.general.visual_dictionary)), size=1)[0])
    results = qa_evaluator.evaluate(probs, strict_mode=True, top_k=5, report=False)
    assert len(results['average']['accuracy']) == 5
    assert results['average']['accuracy'] == [0.3333333, 0.3333333, 0.3333333, 0.3333333, 0.6666667]


def test_qa_evaluate_exclusion(qa_evaluator):
    assert len(qa_evaluator.filtered_qa_data) == 3
    probs = []
    for i in range(len(qa_evaluator.filtered_qa_data)):
        probs.append(np.random.dirichlet(np.ones(
            len(qa_evaluator.visual_qa.general.visual_dictionary)), size=1)[0])
    results = qa_evaluator.evaluate(probs, strict_mode=True, top_k=5, mode='exclusion', report=False)
    assert len(results['average']['accuracy']) == 5


def test_qa_evaluate_inclusion(qa_evaluator):
    assert len(qa_evaluator.filtered_qa_data) == 3
    probs = []
    for i in range(len(qa_evaluator.filtered_qa_data)):
        probs.append(np.random.dirichlet(np.ones(len(
            qa_evaluator.visual_qa.general.visual_dictionary)), size=1)[0])
    results = qa_evaluator.evaluate(probs, strict_mode=True, top_k=5, mode='inclusion', report=False)
    assert len(results['average']['accuracy']) == 5


def test_find_errors_improvements(qa_evaluator):
    qa_evaluator.prior_probabilities = np.array([[0.1428] * 7] * 3)
    labels = [1, 3, 2]
    filtered_probabilities = np.array([[0.1428] * 7] * 3)
    errors = True
    improvements = True
    by_definition_errors, test_matrix_diff_zeros, test_matrix_diff_twos, by_definition_improvements, rank_change = \
        qa_evaluator.find_errors_improvements(labels, filtered_probabilities, errors, improvements)

    assert isinstance(by_definition_errors, list)
    assert isinstance(test_matrix_diff_zeros, list)
    assert isinstance(test_matrix_diff_twos, list)
    assert isinstance(by_definition_improvements, list)
    assert len(rank_change) == 3


def test_find_definition_errors(qa_evaluator):
    df = qa_evaluator.visual_qa.general.knowledge
    i = 0
    filter_groups = qa_evaluator.visual_qa.general.create_filter_groups()
    test_matrix_diff_zeros = test_matrix_diff_twos = []
    actual_test_matrix_diff_zeros, actual_test_matrix_diff_twos = qa_evaluator.find_definition_errors(
        df, i, filter_groups, test_matrix_diff_zeros, test_matrix_diff_twos,
    )

    assert len(actual_test_matrix_diff_zeros) == 0
    assert len(actual_test_matrix_diff_twos) == 0


def test_check_exclusion_errors(qa_evaluator):
    df = qa_evaluator.visual_qa.general.knowledge
    i = 0
    indx = 'AIP:0000733'
    key = 'age'
    value = 'infants_<2'
    def_matrix_zeros_error = qa_evaluator.check_exclusion_errors(df, i, indx, key, value)
    assert len(def_matrix_zeros_error) == 0

    key = 'body_location'
    value = 'nostril'
    def_matrix_zeros_error = qa_evaluator.check_exclusion_errors(df, i, indx, key, value)
    assert len(def_matrix_zeros_error) == 1


def test_check_inclusion_errors(qa_evaluator):
    df = qa_evaluator.visual_qa.general.knowledge
    i = 0
    indx = 'AIP:0000733'
    filter_groups = qa_evaluator.visual_qa.general.create_filter_groups()

    key = 'colour'
    value = 'no_discolouration'
    def_matrix_two_error = qa_evaluator.check_inclusion_errors(df, i, indx, key, value, filter_groups)
    assert len(def_matrix_two_error) == 0

    key = 'colour'
    value = 'yellow'

    def_matrix_two_error = qa_evaluator.check_inclusion_errors(df, i, indx, key, value, filter_groups)
    assert len(def_matrix_two_error) == 5


def test_find_top_k_questions(qa_evaluator):
    qa_evaluator.prior_probabilities = np.array([[0.1428] * 7] * 3)
    top_answers = qa_evaluator.find_top_k_questions(top_k=5, strict_mode=True)
    assert top_answers == ['body_temperature', 'body_location', 'average_size', 'characteristic_tactile',
                           'type_under_1_cm']


def test_find_top_k_questions_sequential(qa_evaluator):
    qa_evaluator.prior_probabilities = np.array([[0.1428] * 7] * 3)
    top_answers = qa_evaluator.find_top_k_questions_sequential(top_k=5, strict_mode=True)
    assert top_answers == ['body_temperature', 'characteristic_borders', 'characteristic_groupings',
                           'characteristic_hair', 'characteristic_shape']
