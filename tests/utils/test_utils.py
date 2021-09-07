import pytest
import jsonschema

from freezegun import freeze_time
from studio.utils import utils


def test_read_json(auto_schema_json):
    json_file = utils.read_json(auto_schema_json)
    assert len(json_file) == 6, "json_file is built out of six sections (`$schema`, `title`, `type`, `properties`, `additionalProperties` and `required`)."


def test_read_yaml(animals_data_directory_yaml):
    yaml_file = utils.read_yaml(animals_data_directory_yaml)
    assert len(yaml_file) == 3, "yaml_file is built out of three section."


def test_mkdir():
    pass


def test_mkdirs():
    pass


def test_ignore_chars():
    pass


@freeze_time("2021-01-01 00:00:00")
def test_timestamp():
    actual = utils.timestamp()
    expected = '2021/01/01_00:00:00'
    assert actual == expected


def test_validate_dummy_schema(animals_data_directory_yaml):
    """ Tests that an expected config file is correctly validated
        through a schema file
    """
    config_file = utils.read_yaml(animals_data_directory_yaml)
    utils.validate_config(config_file, 'auto')


def test_validate_dummy_schema_wrong_key_word(animals_data_directory_yaml):
    """ Tests that the schema of a config file having an unexpected key word
    raises a error
    """
    config_file = utils.read_yaml(animals_data_directory_yaml)

    # Add an unexpected key value in the config file in the `experiment` section
    config_file['experiment']['wrong_key'] = "wrong_value"

    # Check that the schema validator raises an error about the unexpected key value `wrong_key` we added
    expected = "Additional properties are not allowed ('wrong_key' was unexpected)"
    with pytest.raises(jsonschema.exceptions.ValidationError) as e:
        utils.validate_config(config_file, 'auto')
        assert expected in str(e)


def test_get_dir_files():
    pass


def test_get_file_paths_recursive():
    pass
