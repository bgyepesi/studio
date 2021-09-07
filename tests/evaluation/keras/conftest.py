import os
import pytest
import numpy as np
import pandas as pd


from copy import deepcopy
from keras.layers import Dense
from keras.models import Model, Input
from keras_model_specs import ModelSpec
from studio.evaluation.keras import utils
from studio.evaluation.keras.evaluators import CNNEvaluator, CNNTagEvaluator, SequentialCNNEvaluator, VisualQAEvaluator

DIR_TESTS = os.path.join('tests', 'evaluation', 'keras')
DIR_CONFIG = os.path.join(DIR_TESTS, 'files', 'config_files')
DIR_MODELS = os.path.join(DIR_TESTS, 'tmp', 'fixtures', 'models')

first_gate_config = {
    'id': 'first_gate',
    'model_map_json': os.path.abspath(os.path.join(DIR_CONFIG, 'first_gate.json')),
    'squash_classes': [[0, 1], [2], [3]],
    'threshold': {
        'threshold_class': 2,
        'threshold_prob': 0.1,
        'trigger_classes': [1]},
    'pass_through': [2],
    'batch_size': 1
}

catdog_config = {'id': 'fake_catdog',
                 'model_map_json': os.path.abspath(os.path.join(DIR_CONFIG, 'catdog.json'))}

utils.dump_yaml_file(os.path.abspath(os.path.join(DIR_MODELS, 'fake_first_gate', 'first_gate.yaml')),
                     first_gate_config)
utils.dump_yaml_file(os.path.abspath(os.path.join(DIR_MODELS, 'fake_catdog', 'catdog.yaml')),
                     catdog_config)


class FakeModel(Model):
    def __init__(self, **options):
        super(FakeModel, self).__init__(inputs=options['inputs'], outputs=options['outputs'])
        self.options = options
        self.predictions = options['predictions']
        self.count = -1

    def predict(self, **kwargs):
        self.count += 1
        predictions = self.predictions[None, :]
        return predictions[:, self.count, :]

    def predict_generator(self, **kwargs):
        return self.predictions


@pytest.fixture(scope='function')
def fake_model_catdog_mobilenet():
    a = Input(shape=(1,))
    b = Dense(1)(a)
    predictions = np.array([[0.738949, 0.26105103], [0.9817811, 0.01821885],
                            [0.92916, 0.07083996], [0.363611, 0.636389]], dtype=np.float32)
    model = FakeModel(inputs=a, outputs=b, predictions=predictions)
    return model


@pytest.fixture(scope='function')
def fake_model_catdog_mobilenet_after_fg():
    a = Input(shape=(1,))
    b = Dense(1)(a)
    predictions = np.array([[0.738949, 0.26105103], [0.92916, 0.07083996]], dtype=np.float32)
    model = FakeModel(inputs=a, outputs=b, predictions=predictions)
    return model


@pytest.fixture(scope='function')
def fake_model_first_gate():
    a = Input(shape=(1,))
    b = Dense(1)(a)
    predictions = np.array([[0.15, 0.1, 0.3, 0.45], [0.25, 0.3, 0.1, 0.35],
                            [0.05, 0.05, 0.7, 0.2], [0.8, 0.05, 0.1, 0.05]], dtype=np.float32)
    model = FakeModel(inputs=a, outputs=b, predictions=predictions)
    return model


@pytest.fixture(scope='session')
def test_dir_tests():
    return DIR_TESTS


@pytest.fixture(scope='session')
def test_fake_catdog_model():
    return os.path.abspath(
        os.path.join(DIR_MODELS, 'fake_catdog', 'fake_catdog.h5'))


@pytest.fixture(scope='session')
def test_fake_first_gate_model():
    return os.path.abspath(
        os.path.join(DIR_MODELS, 'fake_first_gate', 'fake_first_gate.h5'))


@pytest.fixture(scope='session')
def test_catdog_manifest_path():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test_catdog_manifest.json'))


@pytest.fixture(scope='session')
def test_catdog_manifest_tag_path():
    # This dataframe file contains:
    # - one test image that does not contain the tag `case-id` intentionally
    #   to test that images without a tag are considered as unique cases
    # - one test image which is a dog but with a `class_probabilities` of a cat
    #   to test the case when images of the same case have different labels
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test_catdog_manifest_tag.json'))


@pytest.fixture(scope='session')
def test_catdog_manifest_fail_dataframe():
    return pd.read_json(os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test_catdog_manifest_fail.json')))


@pytest.fixture(scope='session')
def test_catdog_manifest_dataframe():
    return pd.read_json(os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test_catdog_manifest.json')))


@pytest.fixture(scope='session')
def test_single_cat_image_path():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test', 'cat', 'cat-1.jpg'))


@pytest.fixture(scope='session')
def test_catdog_dataset_path():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test'))


@pytest.fixture(scope='session')
def test_catdog_dataset_tag_path():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test_tag'))


@pytest.fixture(scope='session')
def test_animals_dataset_path():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'animals', 'test'))


@pytest.fixture(scope='session')
def test_cat_folder():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test', 'cat'))


@pytest.fixture(scope='session')
def test_dog_folder():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test', 'dog'))


@pytest.fixture(scope='session')
def test_image_path():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test', 'cat', 'cat-1.jpg'))


@pytest.fixture(scope='session')
def test_image_paths_list():
    return [os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test', 'cat', 'cat-1.jpg')),
            os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test', 'dog', 'dog-2.jpg'))]


@pytest.fixture(scope='session')
def test_catdog_ensemble_path():
    return os.path.abspath(os.path.join(DIR_MODELS, 'simple_cnn_2'))


@pytest.fixture(scope='session')
def test_catdog_mobilenet_model():
    return os.path.abspath(
        os.path.join(DIR_MODELS, 'simple_cnn_2', 'model_1', 'model_2_classes.h5'))


@pytest.fixture(scope='session')
def test_catdog_mobilenet_model_spec():
    return os.path.abspath(os.path.join(DIR_MODELS, 'simple_cnn_2', 'model_1', 'model_spec.json'))


@pytest.fixture(scope='session')
def test_animals_ensemble_path():
    return os.path.abspath(os.path.join(DIR_MODELS, 'simple_cnn_5'))


@pytest.fixture(scope='session')
def test_animals_mobilenet_path():
    return os.path.abspath(
        os.path.join(DIR_MODELS, 'simple_cnn_5', 'model_1', 'model_5_classes.h5'))


@pytest.fixture(scope='session')
def test_animals_model_spec_path():
    return os.path.abspath(
        os.path.join(DIR_MODELS, 'simple_cnn_5', 'model_1', 'model_spec.json'))


@pytest.fixture(scope='session')
def test_animals_dictionary_path():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'animals', 'dictionary.json'))


@pytest.fixture(scope='session')
def test_average_results_csv_paths():
    return [os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'results_csv', 'eval_avg_1.csv')),
            os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'results_csv', 'eval_avg_2.csv')),
            os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'results_csv', 'eval_avg_3.csv'))]


@pytest.fixture(scope='session')
def test_individual_results_csv_paths():
    return [os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'results_csv', 'eval_class_1.csv')),
            os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'results_csv', 'eval_class_2.csv')),
            os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'results_csv', 'eval_class_3.csv'))]


@pytest.fixture(scope='session')
def test_results_csv_paths():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'eval'))


@pytest.fixture(scope='session')
def model_spec_mobilenet():
    dataset_mean = [142.69182214, 119.05833338, 106.89884415]
    return ModelSpec.get('mobilenet_v1', preprocess_func='mean_subtraction', preprocess_args=dataset_mean)


@pytest.fixture(scope='function')
def fake_evaluator_catdog_mobilenet(fake_model_catdog_mobilenet):
    evaluator = CNNEvaluator(
        batch_size=1,
        id='catdog-mobilenet.hdf5'
    )
    evaluator.models = [fake_model_catdog_mobilenet]
    evaluator.model_specs = [ModelSpec.get('mobilenet_v1', target_size=[224, 224, 3],
                                           preprocess_func='between_plus_minus_1')]
    return evaluator


@pytest.fixture(scope='function')
def evaluator_catdog_mobilenet(test_catdog_mobilenet_model):
    return CNNEvaluator(
        batch_size=1,
        model_path=test_catdog_mobilenet_model
    )


@pytest.fixture(scope='function')
def cnn_evaluator_tag_catdog_mobilenet(test_catdog_mobilenet_model):
    return CNNTagEvaluator(
        batch_size=1,
        model_path=test_catdog_mobilenet_model
    )


@pytest.fixture(scope='session')
def evaluator_catdog_ensemble(test_catdog_ensemble_path):
    return CNNEvaluator(
        ensemble_models_dir=test_catdog_ensemble_path,
        combination_mode='arithmetic',
        batch_size=1
    )


@pytest.fixture(scope='session')
def evaluator_animals_mobilenet_class_inference(test_animals_mobilenet_path,
                                                test_animals_dictionary_path):
    evaluator = CNNEvaluator(
        model_path=test_animals_mobilenet_path,
        concept_dictionary_path=test_animals_dictionary_path,
        batch_size=1
    )
    return evaluator


@pytest.fixture(scope='session')
def evaluator_animals_ensemble_class_inference(test_animals_ensemble_path,
                                               test_animals_dictionary_path):
    evaluator = CNNEvaluator(
        ensemble_models_dir=test_animals_ensemble_path,
        concept_dictionary_path=test_animals_dictionary_path,
        combination_mode='arithmetic',
        batch_size=1
    )
    return evaluator


@pytest.fixture(scope='session')
def metrics_top_k_binary_class():
    concepts = [{'id': 'class0', 'label': 'class0'}, {'id': 'class1', 'label': 'class1'}]
    y_true = np.asarray([0, 1, 0, 1])  # 4 samples, 2 classes.
    y_probs = np.asarray([[1, 0], [0.2, 0.8], [0.8, 0.2], [0.35, 0.65]])
    return concepts, y_true, y_probs


@pytest.fixture(scope='session')
def metrics_top_k_multi_class():
    concepts = [{'id': 'class0', 'label': 'class0'},
                {'id': 'class1', 'label': 'class1'},
                {'id': 'class3', 'label': 'class3'}]
    y_true = np.asarray([0, 1, 2, 2])  # 4 samples, 3 classes.
    y_probs = np.asarray([[1, 0, 0], [0.2, 0.2, 0.6], [0.8, 0.2, 0], [0.35, 0.25, 0.4]])
    return concepts, y_true, y_probs


@pytest.fixture(scope='session')
def metrics_top_k_multi_class_classes_not_present():
    concepts = [{'id': 'class0', 'label': 'class0'},
                {'id': 'class1', 'label': 'class1'},
                {'id': 'class3', 'label': 'class3'}]
    y_true = np.asarray([0, 0, 1])  # 3 samples, 3 classes.
    y_probs = np.asarray([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
    return concepts, y_true, y_probs


@pytest.fixture(scope='session')
def first_gate_config_yaml():
    return os.path.abspath(os.path.join(DIR_MODELS, 'fake_first_gate', 'first_gate.yaml'))


@pytest.fixture(scope='session')
def catdog_config_yaml():
    return os.path.abspath(os.path.join(DIR_MODELS, 'fake_catdog', 'catdog.yaml'))


@pytest.fixture(scope='function')
def get_catdog_config(catdog_config):
    return deepcopy(catdog_config)


@pytest.fixture(scope='function')
def get_first_gate_config():
    return deepcopy(first_gate_config)


@pytest.fixture(scope='session')
def test_image_paths_list_sequential():
    return np.array([os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test', 'cat', 'cat-1.jpg')),
                     os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test', 'cat', 'cat-4.jpg')),
                     os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test', 'dog', 'dog-2.jpg')),
                     os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog', 'test', 'dog', 'dog-4.jpg'))])


@pytest.fixture(scope='session')
def test_labels_sequential():
    return np.array([['d', 'cat'], ['d', 'cat'], ['d', 'dog'], ['d', 'dog']])


@pytest.fixture(scope='function')
def sequential_cnn_evaluator(first_gate_config_yaml, catdog_config_yaml,
                             fake_model_first_gate, fake_model_catdog_mobilenet_after_fg):

    gates_configuration_path = [first_gate_config_yaml, catdog_config_yaml]
    sequential_cnn_eval = SequentialCNNEvaluator(
        gates_configuration_path=gates_configuration_path
    )

    sequential_cnn_eval.cnn_evaluators[0].models = [fake_model_first_gate]
    sequential_cnn_eval.cnn_evaluators[0].model_specs = [ModelSpec.get('mobilenet_v1', target_size=[224, 224, 3],
                                                                       preprocess_func='between_plus_minus_1')]
    sequential_cnn_eval.cnn_evaluators[1].models = [fake_model_catdog_mobilenet_after_fg]
    sequential_cnn_eval.cnn_evaluators[1].model_specs = [ModelSpec.get('mobilenet_v1', target_size=[224, 224, 3],
                                                                       preprocess_func='between_plus_minus_1')]
    return sequential_cnn_eval


@pytest.fixture(scope='session')
def test_qa_data_json_path():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'qa_files', 'test_qa_data.json'))


@pytest.fixture(scope='session')
def visual_dictionary_json_path():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'qa_files', 'visual_dictionary.json'))


@pytest.fixture(scope='session')
def valid_evidence_json_path():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'qa_files', 'valid_evidence.json'))


@pytest.fixture(scope='session')
def by_definition_csv_file():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'qa_files', 'by_definition_matrix_never_must.csv'))


@pytest.fixture(scope='function')
def qa_evaluator(test_qa_data_json_path, by_definition_csv_file, visual_dictionary_json_path, valid_evidence_json_path):
    return VisualQAEvaluator(report_dir='./evaluations_test/',
                             by_definition_csv=by_definition_csv_file,
                             visual_dictionary=visual_dictionary_json_path,
                             qa_data_json=test_qa_data_json_path,
                             valid_evidence=valid_evidence_json_path
                             )
