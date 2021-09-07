import os

from studio.steps.train import Train
from studio.steps.data import TrainData
from backports.tempfile import TemporaryDirectory


def test_train_on_catdog_datasets_with_directories(animals_data_directory_config):
    # Add absolute paths for CircleCI
    animals_data_directory_config['steps']['train']['data']['input']['train_class_manifest_path'] = \
        os.path.abspath(animals_data_directory_config['steps']['train']['data']['input']['train_class_manifest_path'])
    animals_data_directory_config['steps']['train']['data']['input']['val_class_manifest_path'] = \
        os.path.abspath(animals_data_directory_config['steps']['train']['data']['input']['val_class_manifest_path'])

    with TemporaryDirectory() as output_test_dir:

        train = Train(config=animals_data_directory_config['steps']['train'],
                      output_dir=os.path.abspath(output_test_dir),
                      num_gpus=0)
        train.run()

        # check if the generated files contain the expected files `history.pickle`, `train_acc.txt`, `val_acc.txt`, `train_loss.txt`, `val_loss.txt`, `model_spec.json`, `model_max_acc.hdf5`
        model_architecture = animals_data_directory_config['steps']['train']['settings']['architecture']
        generated_files = os.listdir(os.path.join(os.path.abspath(os.path.join(output_test_dir, 'train')), '{}_1_epochs'.format(model_architecture), 'iter_0'))
        assert set(['history.pickle', 'train_acc.txt', 'val_acc.txt', 'train_loss.txt', 'val_loss.txt', 'model_spec.json', 'model_max_acc.hdf5']).issubset(generated_files)


def test_data_train_steps_on_animals_datasets_with_manifest(animals_data_manifest_train_config):

    with TemporaryDirectory() as output_test_dir:

        # Instantiate the data step
        data = TrainData(config=animals_data_manifest_train_config['steps']['data']['train'], output_dir=output_test_dir)
        train_class_manifest_path, val_class_manifest_path = data.run()

        # update the data `input` with absolute paths for CircleCI
        animals_data_manifest_train_config['steps']['train']['data']['input']['train_class_manifest_path'] = os.path.abspath(train_class_manifest_path)
        animals_data_manifest_train_config['steps']['train']['data']['input']['val_class_manifest_path'] = os.path.abspath(val_class_manifest_path)

        train = Train(config=animals_data_manifest_train_config['steps']['train'],
                      output_dir=os.path.abspath(output_test_dir),
                      num_gpus=0)
        train.run()

        # check if the generated files contain the expected files `history.pickle`, `train_acc.txt`, `val_acc.txt`, `train_loss.txt`, `val_loss.txt`, `model_spec.json`, `model_max_acc.hdf5`
        model_architecture = animals_data_manifest_train_config['steps']['train']['settings']['architecture']
        generated_files = os.listdir(os.path.join(os.path.abspath(os.path.join(output_test_dir, 'train')), '{}_1_epochs'.format(model_architecture), 'iter_0'))
        assert set(['history.pickle', 'train_acc.txt', 'val_acc.txt', 'train_loss.txt', 'val_loss.txt', 'model_spec.json', 'model_max_acc.hdf5']).issubset(generated_files)


def test_train_step_on_animals_datasets(animals_data_manifest_train_config, animals_train_class_manifest, animals_val_class_manifest):

    with TemporaryDirectory() as output_test_dir:

        # update the config file with the absolute paths of the data `input` for CircleCI
        animals_data_manifest_train_config['steps']['train']['data']['input']['train_class_manifest_path'] = os.path.abspath(animals_train_class_manifest)
        animals_data_manifest_train_config['steps']['train']['data']['input']['val_class_manifest_path'] = os.path.abspath(animals_val_class_manifest)

        train = Train(config=animals_data_manifest_train_config['steps']['train'],
                      output_dir=os.path.abspath(output_test_dir),
                      num_gpus=0)
        train.run()

        # check if the generated files contain the expected files `history.pickle`, `train_acc.txt`, `val_acc.txt`, `train_loss.txt`, `val_loss.txt`, `model_spec.json`, `model_max_acc.hdf5`
        model_architecture = animals_data_manifest_train_config['steps']['train']['settings']['architecture']
        generated_files = os.listdir(os.path.join(os.path.abspath(os.path.join(output_test_dir, 'train')), '{}_1_epochs'.format(model_architecture), 'iter_0'))
        assert set(['history.pickle', 'train_acc.txt', 'val_acc.txt', 'train_loss.txt', 'val_loss.txt', 'model_spec.json', 'model_max_acc.hdf5']).issubset(generated_files)


def test_train_get_model_specs(animals_data_directory_config):
    # Add absolute paths for CircleCI
    animals_data_directory_config['steps']['train']['data']['input']['train_class_manifest_path'] = \
        os.path.abspath(animals_data_directory_config['steps']['train']['data']['input']['train_class_manifest_path'])
    animals_data_directory_config['steps']['train']['data']['input']['val_class_manifest_path'] = \
        os.path.abspath(animals_data_directory_config['steps']['train']['data']['input']['val_class_manifest_path'])

    with TemporaryDirectory() as output_test_dir:
        train = Train(config=animals_data_directory_config['steps']['train'],
                      output_dir=os.path.abspath(output_test_dir),
                      num_gpus=0)
        setattr(train, 'train_data_stats', {})
        setattr(train, 'target_size', (299, 299, 3))
        train.train_data_stats['mean'] = [1, 2, 3]
        model_spec = train.get_model_spec()
        assert model_spec.preprocess_func == 'mean_subtraction'

        animals_data_directory_config['steps']['train']['data']['data_processing']['preprocess_func'] = None
        animals_data_directory_config['steps']['train']['data']['data_processing']['subtract_dataset_mean'] = False

        train = Train(config=animals_data_directory_config['steps']['train'],
                      output_dir=os.path.abspath(output_test_dir),
                      num_gpus=0)
        setattr(train, 'train_data_stats', {})
        setattr(train, 'target_size', (299, 299, 3))
        train.train_data_stats['mean'] = [1, 2, 3]
        model_spec = train.get_model_spec()
        assert model_spec.preprocess_func == 'between_plus_minus_1', "matching preprocess_function to mobilenet_v1 one"
        assert model_spec.preprocess_args is None
