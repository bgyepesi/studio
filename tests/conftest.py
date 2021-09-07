# import os
# import json
# import pytest
# import pandas as pd
#
# from studio.utils import utils
# from studio.commands.auto import Auto
# from studio.data.snapshots import Snapshot
# from studio.data.ontology import Ontology
#
#
# @pytest.fixture(scope='session')
# def animals_data_directory_yaml():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'configs', 'animals_data_directory.yaml'))
#
#
# @pytest.fixture(scope='session')
# def test_animals_image_path():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'dataset', 'animals', 'files', '136fSmPwG23Luh3J9dLoVMfG'))
#
#
# @pytest.fixture(scope='session')
# def test_dataset_snapshot_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'dataset_snapshot.json'))
#
#
# @pytest.fixture(scope='session')
# def animals_data_directory_config(animals_data_directory_yaml):
#     config = utils.read_yaml(animals_data_directory_yaml)
#     utils.validate_config(config, 'auto', defaults=True)
#     return config
#
#
# @pytest.fixture(scope='session')
# def aip_data_modelmap_yaml():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'configs', 'aip_data_modelmap.yaml'))
#
#
# @pytest.fixture(scope='session')
# def aip_data_modelmap_config(aip_data_modelmap_yaml):
#     config = utils.read_yaml(aip_data_modelmap_yaml)
#     utils.validate_config(config, 'auto', defaults=True)
#     return config
#
#
# @pytest.fixture(scope='session')
# def animals_data_manifest_train_yaml():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'configs', 'animals_config_data_manifest_train.yaml'))
#
#
# @pytest.fixture(scope='session')
# def animals_data_manifest_eval_yaml():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'configs', 'animals_config_data_manifest_eval.yaml'))
#
#
# @pytest.fixture(scope='session')
# def animals_data_manifest_train_config(animals_data_manifest_train_yaml):
#     config = utils.read_yaml(animals_data_manifest_train_yaml)
#     utils.validate_config(config, 'auto', defaults=True)
#     return config
#
#
# @pytest.fixture(scope='session')
# def animals_data_manifest_eval_config(animals_data_manifest_eval_yaml):
#     config = utils.read_yaml(animals_data_manifest_eval_yaml)
#     utils.validate_config(config, 'auto', defaults=True)
#     return config
#
#
# @pytest.fixture(scope='session')
# def animals_dataset_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'dataset', 'animals', 'manifest.json'))
#
#
# @pytest.fixture(scope='session')
# def animals_train_lab_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'animals_train_lab_manifest.json'))
#
#
# @pytest.fixture(scope='session')
# def animals_val_lab_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'animals_val_lab_manifest.json'))
#
#
# @pytest.fixture(scope='session')
# def animals_test_lab_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'animals_test_lab_manifest.json'))
#
#
# @pytest.fixture(scope='session')
# def animals_train_class_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'animals_train_class_manifest.json'))
#
#
# @pytest.fixture(scope='session')
# def animals_val_class_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'animals_val_class_manifest.json'))
#
#
# @pytest.fixture(scope='session')
# def animals_test_class_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'animals_test_class_manifest.json'))
#
#
# @pytest.fixture(scope='session')
# def dataset_snapshot_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'dataset_snapshot.json'))
#
#
# @pytest.fixture(scope='session')
# def auto_schema_json():
#     return os.path.abspath(os.path.join('studio', 'schemas', 'auto.json'))
#
#
# @pytest.fixture(scope='session')
# def auto_command(animals_data_directory_yaml):
#     return Auto(animals_data_directory_yaml)
#
#
# @pytest.fixture(scope='session')
# def animals_catdog_val_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'dataset', 'animals', 'catdog', 'val', 'manifest.json'))
#
#
# @pytest.fixture(scope='session')
# def animals_catdog_val_path():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'dataset', 'animals', 'catdog', 'val'))
#
#
# @pytest.fixture(scope='session')
# def dummy_ontology():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'dummy_ontology.json'))
#
#
# @pytest.fixture(scope='session')
# def animals_ontology_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'animals_ontology.json'))
#
#
# @pytest.fixture(scope='session')
# def aip_ontology_manifest():
#     return os.path.abspath(os.path.join('data', 'ontology', 'ontology.json'))
#
#
# @pytest.fixture(scope='session')
# def aip_macroscopic_diagnosis_nodes_df():
#     return pd.read_csv(os.path.abspath(os.path.join('data', 'ontology', 'macroscopic_diagnosis_nodes.csv')))
#
#
# @pytest.fixture(scope='session')
# def aip_dermoscopic_diagnosis_nodes_df():
#     return pd.read_csv(os.path.abspath(os.path.join('data', 'ontology', 'dermoscopic_diagnosis_nodes.csv')))
#
#
# @pytest.fixture(scope='session')
# def ontology_manifest_errors_dn(dummy_ontology):
#     with open(dummy_ontology) as f:
#         ontology_manifest = json.load(f)
#     ontology_manifest['nodes'][5]['type'] = Ontology.NODE_TYPE_DIAGNOSIS
#     ontology_manifest['nodes'][8]['type'] = Ontology.NODE_TYPE_DIAGNOSIS
#     return ontology_manifest
#
#
# @pytest.fixture(scope='session')
# def ontology_manifest_errors_malignancy(dummy_ontology):
#     with open(dummy_ontology) as f:
#         ontology_manifest = json.load(f)
#     ontology_manifest['nodes'][9]['malignancy'] = Ontology.MALIGNANCY_BENIGN
#     return ontology_manifest
#
#
# @pytest.fixture(scope='session')
# def ontology_manifest_errors_duplicate_ids(dummy_ontology):
#     with open(dummy_ontology) as f:
#         ontology_manifest = json.load(f)
#     ontology_manifest['nodes'][9]['id'] = 'AIP:0000007'
#     ontology_manifest['nodes'][7]['id'] = 'AIP:0002471'
#     return ontology_manifest
#
#
# @pytest.fixture(scope='session')
# def ontology_manifest_multiple_errors(dummy_ontology):
#     with open(dummy_ontology) as f:
#         ontology_manifest = json.load(f)
#     ontology_manifest['nodes'][9]['malignancy'] = Ontology.MALIGNANCY_BENIGN
#     ontology_manifest['nodes'][5]['type'] = Ontology.NODE_TYPE_DIAGNOSIS
#     ontology_manifest['nodes'][8]['type'] = Ontology.NODE_TYPE_DIAGNOSIS
#     ontology_manifest['nodes'][9]['label'] = 'adnexal disease'
#     ontology_manifest['nodes'][7]['label'] = 'acne vulgaris'
#     return ontology_manifest
#
#
# @pytest.fixture(scope='session')
# def ontology_manifest_errors_duplicate_labels(dummy_ontology):
#     with open(dummy_ontology) as f:
#         ontology_manifest = json.load(f)
#     ontology_manifest['nodes'][9]['label'] = 'adnexal disease'
#     ontology_manifest['nodes'][7]['label'] = 'acne vulgaris'
#     return ontology_manifest
#
#
# @pytest.fixture(scope='session')
# def datasets_ontologies_csv():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'datasets-ontologies.csv'))
#
#
# @pytest.fixture(scope='session')
# def dataset_dataframe(datasets_ontologies_csv):
#     return pd.read_csv(datasets_ontologies_csv, converters={'node_id': lambda x: str(x)})
#
#
# @pytest.fixture(scope='session')
# def snapshot_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'snapshot_manifest.json'))
#
#
# @pytest.fixture(scope='session')
# def animals_one_hot_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'dataset', 'animals', 'manifest_one_hot.json'))
#
#
# @pytest.fixture(scope='session')
# def snapshot_reviews_manifest():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'snapshot_reviews_manifest.json'))
#
#
# @pytest.fixture(scope='session')
# def reviews_json():
#     return os.path.abspath(os.path.join('tests', 'fixtures', 'data', 'test_reviews.json'))
#
#
# @pytest.fixture(scope='function')
# def ontology_tree(dummy_ontology):
#     """
#     root [AIP:root](unspecified)
#         disease [AIP:0000000](unspecified)
#             cutaneous disease [AIP:0000001](unspecified)
#                 adnexal disease [AIP:0000007](unspecified)
#                     acne vulgaris [AIP:0002471](B)*
#                         acne mechanica [AIP:0001379](B)
#                         acne fulminans [AIP:0002478](B)
#                         acne conglobata [AIP:0002480](B)
#                             follicular occlusion tetrad [AIP:0002481](B)
#                         acne excoriee des jeunes filles [AIP:0002484](B)
#                     dummy node [AIP:0100001](M)
#                         acneiform eruption [AIP:0002475](B)
#                             childhood flexural comedones [AIP:0002491](B)*
#     """
#     return Ontology(ontology_manifest=dummy_ontology)
#
#
# @pytest.fixture(scope='function')
# def json_ontology():
#     """Expected JSON ontology values after converting `_create_example_dataframe()` to JSON."""
#     json_nodes = [
#         {
#             'id': 'root_id',
#             'label': 'root_label',
#             'type': 'root_type',
#             'malignancy': 'unspecified',
#             "show_during_review": True
#         },
#         {
#             'id': 'child1_id',
#             'label': 'child1_label',
#             'type': 'child1_type',
#             'malignancy': 'benign',
#             "show_during_review": True
#         },
#         {
#             'id': 'child2_id',
#             'label': 'child2_label',
#             'type': 'child2_type',
#             'malignancy': 'malignant',
#             "show_during_review": False
#         }
#     ]
#
#     json_edges = [
#         {'from': 'root_id', 'to': 'child1_id'},
#         {'from': 'root_id', 'to': 'child2_id'},
#     ]
#
#     out_json_ontology = {Ontology.NODES: json_nodes, Ontology.EDGES: json_edges}
#     return out_json_ontology
#
#
# @pytest.fixture(scope='session')
# def snapshot(snapshot_manifest):
#     return Snapshot(snapshot_manifest=snapshot_manifest)
#
#
# @pytest.fixture(scope='session')
# def class_snapshot(snapshot_class_manifest):
#     return Snapshot(snapshot_manifest=snapshot_class_manifest)
#
#
# @pytest.fixture(scope='session')
# def animals_snapshot_dataframe(animals_one_hot_manifest):
#     return pd.read_json(animals_one_hot_manifest)
#
#
# @pytest.fixture(scope='session')
# def snapshot_with_reviews(snapshot_reviews_manifest):
#     return Snapshot(snapshot_manifest=snapshot_reviews_manifest)
#
#
# @pytest.fixture(scope='session')
# def review_list(reviews_json):
#     with open(reviews_json, 'r') as f:
#         review_list = json.load(f)
#     return review_list
#
#
# @pytest.fixture(scope='session')
# def node_frequency_counts():
#     return pd.DataFrame({'node_id': ['AIP:root',
#                                      'AIP:0002471',
#                                      'AIP:0002481',
#                                      'AIP:0002478',
#                                      'AIP:0002491'],
#                          'frequency': [0, 1, 0, 2, 5]})
