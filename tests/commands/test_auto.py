import os

from studio.utils import utils
from freezegun import freeze_time
from studio.commands.auto import Auto
from backports.tempfile import TemporaryDirectory


@freeze_time("2021-01-01 00:00:00")
def test_get_identifier(auto_command):
    author = 'dummy_author'
    experiment = 'dummy_experiment'
    actual = auto_command.get_identifier(author, experiment)
    expected = '2021_01_01_00-00-00_dummy_author_dummy_experiment'
    assert actual == expected


def test_train_and_eval_steps_on_animals_datasets_with_directories(animals_data_directory_config):

    with TemporaryDirectory() as output_test_dir:
        # Overwrite the experiment's output directory
        animals_data_directory_config['experiment']['output_dir'] = output_test_dir
        # Save the updated yaml file
        updated_yaml_config_path = os.path.join(output_test_dir, 'animals_config_data_directory.yaml')
        utils.store_yaml(animals_data_directory_config, updated_yaml_config_path)
        # Run the `Auto` object
        auto = Auto(os.path.abspath(updated_yaml_config_path))
        auto.run()

        # Check if the data step returns the expected `train_class_manifest_path` and `val_class_manifest_path` files
        assert auto.config['steps']['train']['data']['input']['train_class_manifest_path'] == os.path.join(os.path.join(output_test_dir, auto.experiment_id, 'train'), 'train_class_manifest.json')
        assert auto.config['steps']['train']['data']['input']['val_class_manifest_path'] == os.path.join(os.path.join(output_test_dir, auto.experiment_id, 'train'), 'val_class_manifest.json')

        # For the train step, check if the generated files contain the expected files `history.pickle`, `train_acc.txt`, `val_acc.txt`, `train_loss.txt`, `val_loss.txt`,
        # `model_spec.json`, `model_max_acc.hdf5`
        model_architecture = animals_data_directory_config['steps']['train']['settings']['architecture']
        generated_files = os.listdir(os.path.join(os.path.abspath(os.path.join(output_test_dir, auto.experiment_id, 'train')), '{}_1_epochs'.format(model_architecture), 'iter_0'))
        assert set(['history.pickle', 'train_acc.txt', 'val_acc.txt', 'train_loss.txt', 'val_loss.txt', 'model_spec.json', 'model_max_acc.hdf5']).issubset(generated_files)

        # For the eval step, check if the generated files contain the two expected csv files `catdog_test_average.csv` and `catdog_test_individual.csv`
        generated_files = os.listdir(os.path.abspath(os.path.join(output_test_dir, auto.experiment_id, 'eval')))
        assert set(['{}_average.csv'.format(auto.eval.id), '{}_individual.csv'.format(auto.eval.id)]).issubset(generated_files)


def test_train_step_on_animals_datasets_with_manifest(animals_data_manifest_train_config):

    with TemporaryDirectory() as output_test_dir:
        # Overwrite the experiment's output directory
        animals_data_manifest_train_config['experiment']['output_dir'] = output_test_dir
        # Save the updated yaml file
        updated_yaml_config_path = os.path.join(output_test_dir, 'animals_config_data_manifest_train.yaml')
        utils.store_yaml(animals_data_manifest_train_config, updated_yaml_config_path)
        # Run the `Auto` object
        auto = Auto(os.path.abspath(updated_yaml_config_path))
        auto.run()

        # check if the generated files contain the expected files `history.pickle`, `train_acc.txt`, `val_acc.txt`, `train_loss.txt`, `val_loss.txt`, `model_spec.json`, `model_max_acc.hdf5`
        model_architecture = animals_data_manifest_train_config['steps']['train']['settings']['architecture']
        generated_files = os.listdir(os.path.join(os.path.abspath(os.path.join(output_test_dir, auto.experiment_id, 'train')), '{}_1_epochs'.format(model_architecture), 'iter_0'))
        assert set(['history.pickle', 'train_acc.txt', 'val_acc.txt', 'train_loss.txt', 'val_loss.txt', 'model_spec.json', 'model_max_acc.hdf5']).issubset(generated_files)


def test_eval_step_on_animals_datasets_with_manifest(animals_data_manifest_eval_config):

    with TemporaryDirectory() as output_test_dir:
        # Overwrite the experiment's output directory
        animals_data_manifest_eval_config['experiment']['output_dir'] = output_test_dir
        # Save the updated yaml file
        updated_yaml_config_path = os.path.join(output_test_dir, 'animals_config_data_manifest_eval.yaml')
        utils.store_yaml(animals_data_manifest_eval_config, updated_yaml_config_path)
        # Run the `Auto` object
        auto = Auto(os.path.abspath(updated_yaml_config_path))
        auto.run()

        # For the eval step, check if the generated files contain the two expected csv files `catdog_test_average.csv` and `catdog_test_individual.csv`
        generated_files = os.listdir(os.path.abspath(os.path.join(output_test_dir, auto.experiment_id, 'eval')))
        assert set(['{}_average.csv'.format(auto.eval.id), '{}_individual.csv'.format(auto.eval.id)]).issubset(generated_files)
