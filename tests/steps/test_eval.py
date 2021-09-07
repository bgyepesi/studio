import os

from studio.steps.eval import Eval
from studio.steps.data import EvalData
from backports.tempfile import TemporaryDirectory


def test_eval_on_catdog_datasets_with_directories(animals_data_directory_config):

    with TemporaryDirectory() as output_test_dir:

        # Instantiate the data step
        data = EvalData(config=animals_data_directory_config['steps']['data']['eval'], output_dir=output_test_dir)
        test_class_manifest_path = data.run()

        # update the data `input` with absolute paths for CircleCI
        animals_data_directory_config['steps']['eval']['data']['input']['test_class_manifest_path'] = os.path.abspath(test_class_manifest_path)

        eval = Eval(config=animals_data_directory_config['steps']['eval'],
                    output_dir=os.path.abspath(output_test_dir))
        eval.id = "catdog_test"
        eval.run()

        # check if the generated files contain the two expected csv files `catdog_test_average.csv` and `catdog_test_individual.csv`
        generated_files = os.listdir(os.path.abspath(os.path.join(output_test_dir, 'eval')))
        assert set(['{}_average.csv'.format(eval.id), '{}_individual.csv'.format(eval.id)]).issubset(generated_files)


def test_data_and_eval_on_catdog_datasets_with_manifest(animals_data_manifest_eval_config):

    with TemporaryDirectory() as output_test_dir:

        # Instantiate the data step
        data = EvalData(config=animals_data_manifest_eval_config['steps']['data']['eval'], output_dir=output_test_dir)
        test_manifest_path = data.run()

        # update the data `input` with absolute paths for CircleCI
        animals_data_manifest_eval_config['steps']['eval']['data']['input']['test_class_manifest_path'] = os.path.abspath(test_manifest_path)

        eval = Eval(config=animals_data_manifest_eval_config['steps']['eval'],
                    output_dir=os.path.abspath(output_test_dir))
        eval.id = "catdog_test"
        eval.run()

        # check if the generated files contain the two expected csv files `catdog_test_average.csv` and `catdog_test_individual.csv`
        generated_files = os.listdir(os.path.abspath(os.path.join(output_test_dir, 'eval')))
        assert set(['{}_average.csv'.format(eval.id), '{}_individual.csv'.format(eval.id)]).issubset(generated_files)


def test_eval_on_catdog_datasets(animals_data_manifest_eval_config, animals_test_class_manifest):

    with TemporaryDirectory() as output_test_dir:

        # update the config file with the absolute paths of the data `input` for CircleCI
        animals_data_manifest_eval_config['steps']['eval']['data']['input']['test_class_manifest_path'] = os.path.abspath(animals_test_class_manifest)

        eval = Eval(config=animals_data_manifest_eval_config['steps']['eval'],
                    output_dir=os.path.abspath(output_test_dir))
        eval.id = "catdog_test"
        eval.run()

        # check if the generated files contain the two expected csv files `catdog_test_average.csv` and `catdog_test_individual.csv`
        generated_files = os.listdir(os.path.abspath(os.path.join(output_test_dir, 'eval')))
        assert set(['{}_average.csv'.format(eval.id), '{}_individual.csv'.format(eval.id)]).issubset(generated_files)
