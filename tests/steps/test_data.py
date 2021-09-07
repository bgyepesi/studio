import os
import pandas as pd

from studio.steps.data import TrainData
from backports.tempfile import TemporaryDirectory


def test_train_data_directory_on_catdog_datasets(animals_data_directory_config):
    # Add absolute paths for CircleCI
    animals_data_directory_config['steps']['data']['train']['directory']['train_dir'] = \
        os.path.abspath(animals_data_directory_config['steps']['data']['train']['directory']['train_dir'])
    animals_data_directory_config['steps']['data']['train']['directory']['val_dir'] = \
        os.path.abspath(animals_data_directory_config['steps']['data']['train']['directory']['val_dir'])

    with TemporaryDirectory() as output_test_dir:

        data = TrainData(config=animals_data_directory_config['steps']['data']['train'], output_dir=os.path.abspath(output_test_dir))
        train_class_manifest_path, val_class_manifest_path = data.run()
        # Check if the data object returns the expected `train_class_manifest_path` and `val_class_manifest_path` files
        assert train_class_manifest_path == os.path.join(os.path.join(output_test_dir, 'train'), 'train_class_manifest.json')
        assert val_class_manifest_path == os.path.join(os.path.join(output_test_dir, 'train'), 'val_class_manifest.json')


def test_train_data_modelmap_on_animals_dataset(aip_data_modelmap_config):
    # Add absolute paths for CircleCI
    aip_data_modelmap_config['steps']['data']['train']['modelmap']['data_directory'] = \
        os.path.abspath(aip_data_modelmap_config['steps']['data']['train']['modelmap']['data_directory'])

    aip_data_modelmap_config['steps']['data']['train']['modelmap']['dataset_manifest_path'] = \
        os.path.abspath(aip_data_modelmap_config['steps']['data']['train']['modelmap']['dataset_manifest_path'])

    aip_data_modelmap_config['steps']['data']['train']['modelmap']['conditions_manifest_path'] = \
        os.path.abspath(aip_data_modelmap_config['steps']['data']['train']['modelmap']['conditions_manifest_path'])

    with TemporaryDirectory() as output_test_dir:
        data = TrainData(config=aip_data_modelmap_config['steps']['data']['train'], output_dir=os.path.abspath(output_test_dir))
        train_manifest_path, val_manifest_path = data.run()

        # Check if expected files exists
        assert os.path.isfile(train_manifest_path)
        assert os.path.isfile(val_manifest_path)

        # Check expected files' values
        train_df = pd.read_json(train_manifest_path)
        val_df = pd.read_json(val_manifest_path)
        assert len(train_df) == 51
        assert len(val_df) == 5


def test_train_data_manifest_on_catdog_datasets(animals_data_manifest_train_config):
    # Add absolute paths for CircleCI
    animals_data_manifest_train_config['steps']['data']['train']['lab']['manifest']['train_lab_manifest_path'] = \
        os.path.abspath(animals_data_manifest_train_config['steps']['data']['train']['lab']['manifest']['train_lab_manifest_path'])

    with TemporaryDirectory() as output_test_dir:

        data = TrainData(config=animals_data_manifest_train_config['steps']['data']['train'], output_dir=output_test_dir)
        train_class_manifest_path, val_class_manifest_path = data.run()

        # Check if the data object returns the expected `train_class_manifest_path` and `val_class_manifest_path` files
        assert train_class_manifest_path == os.path.join(os.path.join(output_test_dir, 'train'), 'train_class_manifest.json')
        assert val_class_manifest_path == os.path.join(os.path.join(output_test_dir, 'train'), 'val_class_manifest.json')
