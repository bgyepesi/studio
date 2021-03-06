import os
import copy
import json
import keras
import platform
import tensorflow
import numpy as np
import pandas as pd

from six import string_types
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras_model_specs import ModelSpec
from studio.training.keras.parallel import make_parallel
from studio.training.keras.callbacks import SensitivityCallback
from studio.training.keras.regularizations import set_model_regularization
from studio.training.keras.data_generators import EnhancedImageDataGenerator


class Trainer(object):

    OPTIONS = {
        'model_spec': {'type': str},
        'output_logs_dir': {'type': str},
        'output_model_dir': {'type': str},
        'activation': {'type': str, 'default': 'softmax'},
        'batch_size': {'type': int, 'default': 1},
        'callback_list': {'type': list, 'default': []},
        'checkpoint_path': {'type': str, 'default': None},
        'class_weights': {'type': None, 'default': None},
        'custom_crop': {'type': bool, 'default': False},
        'custom_model': {'type': None, 'default': None},
        'decay': {'type': float, 'default': 0.0005},
        'dropout_rate': {'type': float, 'default': 0.0},
        'epochs': {'type': int, 'default': 1},
        'freeze_layers_list': {'type': None, 'default': None},
        'loss_function': {'type': str, 'default': 'categorical_crossentropy'},
        'include_top': {'type': bool, 'default': False},
        'input_shape': {'type': None, 'default': None},
        'iterator_mode': {'type': str, 'default': None},
        'loss_weights': {'type': None, 'default': None},
        'max_queue_size': {'type': int, 'default': 16},
        'metrics': {'type': list, 'default': ['accuracy']},
        'model_kwargs': {'type': dict, 'default': {}},
        'momentum': {'type': float, 'default': 0.9},
        'num_classes': {'type': int, 'default': None},
        'num_gpus': {'type': int, 'default': 0},
        'optimizer': {'type': None, 'default': None},
        'pooling': {'type': str, 'default': 'avg'},
        'random_crop_size': {'type': float, 'default': None},
        'regularization_function': {'type': None, 'default': None},
        'regularization_layers': {'type': None, 'default': None},
        'regularize_bias': {'type': bool, 'default': False},
        'save_training_options': {'type': bool, 'default': True},
        'sgd_lr': {'type': float, 'default': 0.01},
        'top_layers': {'type': None, 'default': None},
        'track_sensitivity': {'type': bool, 'default': False},
        'train_data_generator': {'type': None, 'default': None},
        'train_dataset_dataframe_path': {'type': str, 'default': None},
        'train_dataset_dir': {'type': str, 'default': None},
        'train_generator': {'type': None, 'default': None},
        'val_data_generator': {'type': None, 'default': None},
        'val_dataset_dataframe_path': {'type': str, 'default': None},
        'val_dataset_dir': {'type': str, 'default': None},
        'val_generator': {'type': None, 'default': None},
        'verbose': {'type': bool, 'default': False},
        'weights': {'type': str, 'default': 'imagenet'},
        'workers': {'type': int, 'default': 1},
    }

    def __init__(self, **options):
        for key, option in self.OPTIONS.items():
            if key not in options and 'default' not in option:
                raise ValueError('missing required option: %s' % (key, ))
            value = options.get(key, copy.copy(option.get('default')))
            setattr(self, key, value)

        extra_options = set(options.keys()) - set(self.OPTIONS.keys())
        if len(extra_options) > 0:
            raise ValueError('unsupported options given: %s' % (', '.join(extra_options), ))

        if isinstance(self.model_spec, string_types):
            self.model_spec = ModelSpec.get(self.model_spec)
        elif isinstance(self.model_spec, dict):
            self.model_spec = ModelSpec.get(self.model_spec['name'], **self.model_spec)

        options = dict([(key, getattr(self, key)) for key in self.OPTIONS.keys() if getattr(self, key) is not None])
        options['model_spec'] = self.model_spec.as_json()
        self.context = {
            'versions': {
                'python': platform.python_version(),
                'tensorflow': tensorflow.__version__,
                'keras': keras.__version__
            },
            'options': options
        }

        if isinstance(self.loss_function, list) and len(self.loss_function) > 1:
            self.n_outputs = len(self.loss_function)
        else:
            self.n_outputs = 1

        # Set up the training data generator.
        print('Training data')  # To complement Keras message

        self.train_data_generator = self.train_data_generator or EnhancedImageDataGenerator(
            custom_crop=self.custom_crop,
            random_crop_size=self.random_crop_size,
            rotation_range=180,
            width_shift_range=0,
            height_shift_range=0,
            preprocessing_function=self.model_spec.preprocess_input,
            shear_range=0,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

        if not self.train_generator:
            if self.train_dataset_dir is None or not os.path.isdir(self.train_dataset_dir):
                raise ValueError('`train_dataset_dir` must be a valid directory')

            if self.train_dataset_dataframe_path is None:
                self.train_generator = self.train_data_generator.flow_from_directory(
                    self.train_dataset_dir,
                    iterator_mode=self.iterator_mode,
                    n_outputs=self.n_outputs,
                    batch_size=self.batch_size,
                    target_size=self.model_spec.target_size[:2],
                    class_mode='categorical'
                )

            else:
                if self.train_dataset_dataframe_path.endswith('json'):
                    self.train_dataset_dataframe = pd.read_json(self.train_dataset_dataframe_path)
                elif self.train_dataset_dataframe_path.endswith('csv'):
                    self.train_dataset_dataframe = pd.read_csv(self.train_dataset_dataframe_path)
                else:
                    raise ValueError('`train_dataset_dataframe` a json or a csv valid path')
                # We assume the dataframe will have the labels in 'class_probabilities' column and
                # filenames under 'filename' column
                self.train_generator = self.train_data_generator.flow_from_dataframe(
                    self.train_dataset_dataframe,
                    directory=self.train_dataset_dir,
                    batch_size=self.batch_size,
                    x_col="filename",
                    y_col="class_probabilities",
                    iterator_mode=self.iterator_mode,
                    n_outputs=self.n_outputs,
                    target_size=self.model_spec.target_size[:2],
                    class_mode='probabilistic'
                )

            self.num_classes = self.num_classes or self.train_generator.num_classes

        if self.num_classes is None and self.top_layers is None:
            raise ValueError('num_classes must be set to use a custom train_generator with the default fully connected '
                             '+ softmax top_layers')

        # Set up the validation data generator.
        print('Validation data')  # To complement Keras message

        self.val_data_generator = self.val_data_generator or EnhancedImageDataGenerator(
            preprocessing_function=self.model_spec.preprocess_input,
        )

        if not self.val_generator:

            if self.val_dataset_dir is None or not os.path.isdir(self.val_dataset_dir):
                raise ValueError('`val_dataset_dir` must be a valid directory')

            if self.val_dataset_dataframe_path is None:
                self.val_generator = self.val_data_generator.flow_from_directory(
                    self.val_dataset_dir,
                    n_outputs=self.n_outputs,
                    batch_size=self.batch_size,
                    target_size=self.model_spec.target_size[:2],
                    class_mode='categorical',
                    shuffle=False
                )
            else:
                if self.val_dataset_dataframe_path.endswith('json'):
                    self.val_dataset_dataframe = pd.read_json(self.val_dataset_dataframe_path)
                elif self.val_dataset_dataframe_path.endswith('csv'):
                    self.val_dataset_dataframe = pd.read_csv(self.val_dataset_dataframe_path)
                else:
                    raise ValueError('`val_dataset_dataframe` must be a json, a csv or a DataFrame object')
                # We assume the dataframe will have the labels in 'class_probabilities' column and
                # filenames under 'filename' column
                self.val_generator = self.val_data_generator.flow_from_dataframe(
                    self.val_dataset_dataframe,
                    directory=self.val_dataset_dir,
                    batch_size=self.batch_size,
                    x_col="filename",
                    y_col="class_probabilities",
                    n_outputs=self.n_outputs,
                    target_size=self.model_spec.target_size[:2],
                    class_mode='probabilistic'
                )

        # Load a keras model from a checkpoint
        if self.checkpoint_path is not None:
            self.model = load_model(self.checkpoint_path)
        # Load a custom model (not supported by keras-model-specs)
        elif self.custom_model is not None and self.model_spec.model is None:
            self.model = self.custom_model
        # Load a model supported by keras-model-specs
        else:
            self.model = self.model_spec.model(
                input_shape=self.input_shape or self.model_spec.target_size,
                weights=self.weights,
                include_top=self.include_top,
                pooling=self.pooling,
                **self.model_spec.keras_kwargs
            )
            # If top layers are given include them, else include a Dense Layer with Softmax/Sigmoid
            if self.top_layers is None:
                # Init list of layers
                self.top_layers = []

                # Include Dropout (optional if dropout_rate entered as parameter)
                if self.dropout_rate > 0.0:
                    self.top_layers.append(Dropout(self.dropout_rate))

                # Set Dense Layer
                self.top_layers.append(Dense(self.num_classes, name='dense'))

                # Set Activation Layer
                if self.activation == 'sigmoid':
                    self.top_layers.append(Activation('sigmoid', name='act_sigmoid'))
                elif self.activation == 'softmax':
                    self.top_layers.append(Activation('softmax', name='act_softmax'))

            # Layer Assembling
            for i, layer in enumerate(self.top_layers):
                if i == 0:
                    self.top_layers[i] = layer(self.model.output)
                else:
                    self.top_layers[i] = layer(self.top_layers[i - 1])

            # Final Model (last item of self.top_layer contains all of them assembled)
            self.model = Model(self.model.input, self.top_layers[-1])

        # Freeze layers if contained in list
        if self.freeze_layers_list is not None:
            for layer in self.freeze_layers_list:
                if isinstance(layer, int) or isinstance(layer, np.int32) or isinstance(layer, np.int64):
                    self.model.layers[layer].trainable = False
                elif isinstance(layer, str):
                    self.model.get_layer(layer).trainable = False
                else:
                    raise ValueError("%s layer type not supported to freeze layers, we expect an int giving the layer "
                                     "index or a str containing the name of the layer." % (type(layer)))

        if self.regularization_function is not None:
            self.model = set_model_regularization(self.model, self.regularization_function,
                                                  self.regularization_layers, self.regularize_bias)

        # Print the model summary.
        if self.verbose:
            self.model.summary()

        # If num_gpus is higher than one, we parallelize the model
        if self.num_gpus > 1:
            self.model = make_parallel(self.model, self.num_gpus)

        # Override the optimizer or use the default.
        self.optimizer = self.optimizer or optimizers.SGD(
            lr=self.sgd_lr,
            decay=self.decay,
            momentum=self.momentum,
            nesterov=True
        )

        if not os.path.exists(self.output_model_dir):
            os.makedirs(self.output_model_dir)

    def run(self):
        if isinstance(self.loss_function, list) and len(self.loss_function) > 1:
            monitor_acc = 'val_' + self.model.layers[-1].name + '_accuracy'
        else:
            monitor_acc = 'val_accuracy'
        # Set Checkpoint to save the model with the highest accuracy
        checkpoint_acc = ModelCheckpoint(
            os.path.join(self.output_model_dir, 'model_max_acc.hdf5'),
            verbose=1,
            monitor=monitor_acc,
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        )
        self.callback_list.append(checkpoint_acc)

        # Set Checkpoint to save the model with the minimum loss
        checkpoint_loss = ModelCheckpoint(
            os.path.join(self.output_model_dir, 'model_min_loss.hdf5'),
            verbose=1,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min'
        )
        self.callback_list.append(checkpoint_loss)

        # Add sensitivity callback
        if self.track_sensitivity:
            sensitivity_callback = SensitivityCallback(self.val_generator,
                                                       output_model_dir=self.output_model_dir,
                                                       batch_size=self.batch_size)
            self.callback_list.append(sensitivity_callback)

        # Set Tensorboard Visualization
        tensorboard = TensorBoard(
            log_dir=self.output_logs_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )
        tensorboard.set_model(self.model)
        self.callback_list.append(tensorboard)

        # Compile the model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=self.metrics,
            loss_weights=self.loss_weights
        )

        if self.train_generator.samples // self.batch_size == 0:
            raise ValueError('Batch size is higher than the total number of training samples')

        if self.val_generator.samples // self.batch_size == 0:
            raise ValueError('Batch size is higher than the total number of validation samples')

        # Model training
        self.history = self.model.fit_generator(
            self.train_generator,
            verbose=1,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=self.epochs,
            callbacks=self.callback_list,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.samples // self.batch_size,
            workers=self.workers,
            class_weight=self.class_weights,
            max_queue_size=self.max_queue_size
        )

        # Save model at last epoch
        self.model.save(os.path.join(self.output_model_dir, 'final_model.hdf5'))

        # Dump model_spec.json file
        with open(os.path.join(self.output_model_dir, 'model_spec.json'), 'w') as file:
            file.write(json.dumps(self.model_spec.as_json(), indent=True, sort_keys=True))

        # Save training options
        if self.save_training_options:
            with open(os.path.join(self.output_model_dir, 'training_options.json'), 'w') as file:
                safe_options = {}
                for key, value in self.context['options'].items():
                    if value is None:
                        continue
                    try:
                        json.dumps(value)
                        safe_options[key] = value
                    except TypeError:
                        continue
                self.context['options'] = safe_options
                file.write(json.dumps(self.context, indent=True, sort_keys=True))
