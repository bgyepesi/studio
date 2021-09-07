import os
import keras
import pytest


DIR_TESTS = os.path.join('tests', 'training', 'keras')


@pytest.fixture(scope='session')
def files_path_catdog():
    return os.path.abspath(os.path.join(DIR_TESTS, 'files', 'catdog'))


@pytest.fixture(scope='session')
def train_catdog_dataset_path(files_path_catdog):
    return os.path.abspath(os.path.join(files_path_catdog, 'train'))


@pytest.fixture(scope='session')
def train_single_cat_image_path(files_path_catdog):
    return os.path.abspath(os.path.join(files_path_catdog, 'train', 'cat', 'cat-1.jpg'))


@pytest.fixture(scope='session')
def train_catdog_dataset_json_path(files_path_catdog):
    return os.path.abspath(os.path.join(files_path_catdog, 'train_data.json'))


@pytest.fixture(scope='session')
def val_catdog_dataset_path(files_path_catdog):
    return os.path.abspath(os.path.join(files_path_catdog, 'val'))


@pytest.fixture(scope='session')
def val_catdog_dataset_json_path(files_path_catdog):
    return os.path.abspath(os.path.join(files_path_catdog, 'val_data.json'))


# Model will be shared during tests
@pytest.fixture(scope='session')
def simple_model():
    input_layer = keras.layers.Input(shape=(224, 224, 3))
    model = keras.layers.Conv2D(3, (3, 3))(input_layer)
    model = keras.layers.GlobalAveragePooling2D()(model)
    model = keras.models.Model(input_layer, model)

    top_layers = [keras.layers.Dense(2, name='dense'),
                  keras.layers.Activation('softmax', name='act_softmax')]

    # Layer Assembling
    for i, layer in enumerate(top_layers):
        if i == 0:
            top_layers[i] = layer(model.output)
        else:
            top_layers[i] = layer(top_layers[i - 1])

    # Final Model (last item of self.top_layer contains all of them assembled)
    return keras.models.Model(model.input, top_layers[-1])
