# import os
# import pytest
#
# from studio.utils import utils
#
#
# @pytest.fixture('session')
# def example_auto_config_yaml():
#     return os.path.abspath(os.path.join('examples', 'auto.yaml'))
#
#
# @pytest.fixture('session')
# def example_7_way_train_config_yaml():
#     return os.path.abspath(os.path.join('examples', '7_way_train.yaml'))
#
#
# @pytest.fixture('session')
# def example_7_way_eval_config_yaml():
#     return os.path.abspath(os.path.join('examples', '7_way_eval.yaml'))
#
#
# @pytest.fixture('session')
# def example_7_way_train_eval_config_yaml():
#     return os.path.abspath(os.path.join('examples', '7_way_train_eval.yaml'))
#
#
# @pytest.fixture('session')
# def example_tuner_config_yaml():
#     return os.path.abspath(os.path.join('examples', 'tuner.yaml'))
#
#
# def test_auto_example_schema(example_auto_config_yaml):
#     """ This function validates the `auto.yaml` example."""
#     config = utils.read_yaml(example_auto_config_yaml)
#     utils.validate_config(config, 'auto')
#
#
# def test_7_way_train_example_schema(example_7_way_train_config_yaml):
#     """ This function validates the `auto.yaml` example."""
#     config = utils.read_yaml(example_7_way_train_config_yaml)
#     utils.validate_config(config, 'auto')
#
#
# def test_7_way_eval_example_schema(example_7_way_eval_config_yaml):
#     """ This function validates the `auto.yaml` example."""
#     config = utils.read_yaml(example_7_way_eval_config_yaml)
#     utils.validate_config(config, 'auto')
#
#
# def test_7_way_train_eval_example_schema(example_7_way_train_eval_config_yaml):
#     """ This function validates the `auto.yaml` example."""
#     config = utils.read_yaml(example_7_way_train_eval_config_yaml)
#     utils.validate_config(config, 'auto')
#
#
# # def test_tuner_example_schema(example_tuner_config):
# #    """ This function validates the `tuner.yaml` example."""
# #    example_tuner_config = utils.read_yaml(example_tuner_config_yaml)
# #    utils.validate_config(example_tuner_config, 'tuner')
