import os
import yaml
import json
import datetime
import jsonschema
import pandas as pd

from jsonschema import validators, Draft7Validator


def read_json(json_path):
    try:
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        return json_file
    except FileNotFoundError:
        pass


def store_json(dict_file, json_path):
    with open(json_path, 'w') as f:
        file = json.dumps(dict_file, indent=4)
        f.write(file)


def store_yaml(dict_file, yaml_path):
    with open(yaml_path, 'w') as f:
        yaml.dump(dict_file, f, default_flow_style=False)


def read_csv(csv_path):
    try:
        with open(csv_path, 'r') as f:
            df = pd.read_csv(f)
        return df
    except FileNotFoundError:
        pass


def read_yaml(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            yaml_file = yaml.load(f, Loader=yaml.Loader)
        return yaml_file
    except FileNotFoundError:
        pass


def mkdir(path):
    if not os.path.exists(path):
        return os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def path_exists(path):
    if os.path.exists(path):
        return True
    else:
        raise ValueError('Path provided does not exist.')


def ignore_chars(f_name):
    ch = ['.']
    if f_name[0] in ch:
        return True

    return False


def timestamp():
    """
    Returns current timestamp string with format YYYY/MM/DD_HH:MM:SS
    """
    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.strftime('%Y/%m/%d_%X')
    return timestamp


def read_schema(schema_name):
    with open(os.path.normpath(os.path.join(
            os.path.dirname(__file__), '..', 'schemas',
            schema_name + '.json'
    ))) as schema:
        return json.load(schema)


def traverse(object, path, tunables, key):
    """
    Traverse the dictionary object recursively and find every tunabls' path / value pairs.
    """
    if isinstance(object, dict):
        for k, v in object.items():
            if isinstance(v, dict):
                traverse(v, path + "." + k, tunables, key)
            elif isinstance(v, list):
                traverse(v, path + "." + k, tunables, key)
            else:
                if key in path:
                    key_path = path.split(".")
                    if (k == "default") and (v is True):
                        tunables.append(key_path[key_path.index(key) - 2])
    return tunables


def validate_config(instance, schema_name, defaults=True):
    # Validate `instance` with `schema_name` schema values.
    # If defaults set to True, `instance` will be initialized with default values from `schema`.
    with open(os.path.normpath(os.path.join(
            os.path.dirname(__file__), '..', 'schemas',
            schema_name + '.json'
    ))) as schema:
        if defaults:
            default_validator = extend_schema_with_default(Draft7Validator)
            try:
                default_validator(json.load(schema)).validate(instance)
            except ValueError:
                raise ValueError("Error when validating the default schema.")
        else:
            try:
                jsonschema.validate(instance, json.load(schema))
            except ValueError:
                raise ValueError("Error when validating the schema.")


def extend_schema_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property_, subschema in properties.items():
            if "default" in subschema and not isinstance(instance, list):
                instance.setdefault(property_, subschema["default"])

        for error in validate_properties(
            validator, properties, instance, schema,
        ):
            yield error

    return validators.extend(
        validator_class, {"properties": set_defaults},
    )


def get_dir_files(dir_path, ignore_rules_function=ignore_chars, include_orig_path=True):
    """
    Return the names of the files and directories in `dir_path`.
    Args:
        dir_path: a string of the directory to get the files from.
        ignore_rules_function: a function that takes in a string,
            and returns True if should ignore this file, else False.
        include_orig_path: a boolean where if,
            True = the file names with the full path are returned,
            False = only the file names, and not the path, are returned.
    Returns:
        dir_files: a list of the files in this directory.
    """

    if os.path.isdir(dir_path):
        all_dir = os.listdir(dir_path)
        dir_files = []

        for filename in all_dir:
            # filename = filename.strip()
            if ignore_rules_function(filename):
                # If the first char of the filename is in the ignore_chars list, then ignore this file.
                pass
            else:
                if include_orig_path:
                    f_name = os.path.join(dir_path, filename)
                else:
                    f_name = filename

                dir_files.append(f_name)

        return dir_files

    else:
        raise ValueError('dir_path specified do not exist')


def get_file_paths_recursive(file_dir, ext=[".jpg", ".jpeg", ".gif", ".png", ".bmp"],
                             ignore_rules_function=ignore_chars):
    """
    Return all the files within `file_dir` recursively, that match the file extension in `ext`.
    Args:
        file_dir: a string indicating the directory to start from.
        ext: a list of file extensions. If a file has this extension, then return it.
        ignore_rules_function:
    Returns:
        all_file_paths: a list of all the files with an extension in `ext` that are under `file_dir`.
    """

    filename, file_extension = os.path.splitext(file_dir)

    if file_extension.lower() in ext:
        if not ignore_rules_function(file_dir):
            # This is a file we want.
            return [file_dir]

    if os.path.isdir(file_dir):
        # This is a directory. So get all the files, are repeat for each file.
        all_file_paths = []
        sub_files = get_dir_files(file_dir, include_orig_path=True)
        for sub_file in sub_files:
            # Recursive call.
            file_paths = get_file_paths_recursive(sub_file, ext=ext, ignore_rules_function=ignore_rules_function)
            if file_paths is not None:
                all_file_paths = all_file_paths + file_paths

    return sorted(all_file_paths)
