import os
import keras
import pprint
import pickle
import platform
import tensorflow
import pandas as pd
import studio.training.keras
import matplotlib as mpl

from PIL import ImageFile
from keras import optimizers
from studio.utils import train_utils
from keras_model_specs import ModelSpec
from studio.utils import utils, data_utils
from studio.utils.train_utils import CyclicLR, unparallelize
from studio.training.keras.data_generators import EnhancedImageDataGenerator


mpl.use('Agg')

# Solve High Resolution truncated files
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Train(object):
    def __init__(self, config, output_dir, num_gpus):
        self.config = config
        self.output_dir = os.path.join(output_dir, 'train')
        if not os.path.isdir(self.output_dir):
            utils.mkdir(self.output_dir)
        self.num_gpus = num_gpus

    def get_class_weights(self):
        # Return class weights
        if self.config['data']['data_processing']['class_weights']:
            if isinstance(self.config['data']['data_processing']['class_weights_value'], list):
                class_weights = self.config['data']['data_processing']['class_weights_value']
            else:
                print('Creating Class Weights...')
                if self.train_data_stats['class_histogram'] is None:
                    raise ValueError('class_histogram from data specifications is missing.')
                if isinstance(self.config['data']['data_processing']['class_weight_smoothing'], float):
                    print("Smoothing Class Weights with mu = ", self.config['data']['data_processing']['class_weight_smoothing'])
                    class_weights = data_utils.create_class_weights(self.train_data_stats['class_histogram'],
                                                                    smooth=True,
                                                                    mu=self.config['data']['data_processing']['class_weight_smoothing'])
                else:
                    class_weights = data_utils.create_class_weights(self.train_data_stats['class_histogram'],
                                                                    smooth=False)
                print('Class Weights: ', class_weights)
        else:
            class_weights = None

        return class_weights

    def get_model_spec(self):
        # Return Keras Model specifications
        if isinstance(self.config['data']['data_processing']['preprocess_func'], str):
            if self.config['data']['data_processing']['subtract_dataset_mean']:
                # Convert to list, to save correctly JSON in keras-trainer
                preprocess_args = list(self.train_data_stats['mean'])
                model_spec = ModelSpec.get(self.config['settings']['architecture'],
                                           preprocess_func=self.config['data']['data_processing']['preprocess_func'],
                                           preprocess_args=preprocess_args,
                                           target_size=(self.target_size, self.target_size, 3))
            else:
                model_spec = ModelSpec.get(self.config['settings']['architecture'],
                                           preprocess_func=self.config['data']['data_processing']['preprocess_func'],
                                           target_size=(self.target_size, self.target_size, 3))
        # Use the default specified in ModelSpec
        else:
            model_spec = ModelSpec.get(self.config['settings']['architecture'],
                                       target_size=(self.target_size, self.target_size, 3))
        return model_spec

    def freeze_layers(self):
        # Freeze layers
        start_freeze_layer = self.config['hyperparameters']['freeze_layers']['start']
        end_freeze_layer = self.config['hyperparameters']['freeze_layers']['end']

        if isinstance(start_freeze_layer, int) and isinstance(end_freeze_layer, int):
            layers_to_freeze = range(start_freeze_layer, end_freeze_layer)
        else:
            layers_to_freeze = None
        return layers_to_freeze

    def customize_model_output_dir(self, num_epochs=True, lr_sgd=True, batch_size=False, output_path=''):
        # By default, model_output_dir contains 'architecture'
        directory_items = list()
        architecture = self.config['settings']['architecture']
        directory_items.append(architecture)

        if num_epochs:
            key = 'epochs'
            value = str(self.config['hyperparameters']['num_epochs'])
            directory_items.append('_'.join([value, key]))
        elif lr_sgd:
            key = 'lr_sgd'
            value = str(self.config['optimizer']['lr_sgd'])
            directory_items.append('_'.join([value, key]))
        elif batch_size:
            key = 'batch_size'
            value = str(self.data.batch_size)
            directory_items.append('_'.join([value, key]))

        model_output_dir = os.path.join(output_path, '_'.join(directory_items))

        return model_output_dir

    def get_optimizer(self):
        if self.config['optimizer'].get('SGD'):
            optimizer = optimizers.SGD(
                lr=self.config['optimizer']['SGD']['lr'],
                decay=self.config['optimizer']['SGD']['decay'],
                momentum=self.config['optimizer']['SGD']['momentum'],
                nesterov=True
            )
        elif self.config['optimizer'].get('Adam'):
            optimizer = optimizers.Adam(
                lr=self.config['optimizer']['Adam']['lr'],
                beta_1=self.config['optimizer']['Adam']['beta_1'],
                beta_2=self.config['optimizer']['Adam']['beta_2'],
                epsilon=self.config['optimizer']['Adam']['epsilon'],
                decay=self.config['optimizer']['Adam']['decay']
            )
        return optimizer

    def get_cyclical_learning_rates(self):
        mode = self.config['hyperparameters']['cyclical_learning_rate']['mode']
        base_lr = self.config['hyperparameters']['cyclical_learning_rate']['base_lr']
        max_lr = self.config['hyperparameters']['cyclical_learning_rate']['max_lr']

        # Step size value recommended 2-10 times the number of iterations in an epoch
        num_images = self.data.train_data_stats['num_images']
        batch_size = self.data.batch_size
        step_size = self.config['hyperparameters']['cyclical_learning_rate']['step_size']
        step_size = (num_images // batch_size) * step_size

        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step_size, mode=mode)
        return clr

    def get_learning_rate_scheduler(self):
        scheduler_frequency = self.config['optimizer']['SGD']['scheduler_frequency']
        gamma = self.config['optimizer']['SGD']['gamma']
        schedule_lr = train_utils.lr_scheduler(scheduler_frequency, gamma)
        return schedule_lr

    def process_callbacks(self):
        callback_list = []

        # Process Learning Rates
        if self.config['hyperparameters'].get('cyclical_learning_rate'):
            clr = self.get_cyclical_learning_rates()
            callback_list.append(clr)
        elif self.config['optimizer'].get('SGD'):
            schedule_lr = self.get_learning_rate_scheduler()
            callback_list.append(schedule_lr)

        return callback_list

    @staticmethod
    def save_model_history(model, output_dir):
        # Save model history
        history = model.history
        with open(output_dir + '/history.pickle', 'wb') as history_file:
            pickle.dump(history, history_file)

        # Accuracy metrics
        with open(output_dir + '/train_acc.txt', 'w+') as history_file:
            for acc in history.history['accuracy']:
                history_file.write(str(acc) + '\n')

        with open(output_dir + '/val_acc.txt', 'w+') as history_file:
            for acc in history.history['val_accuracy']:
                history_file.write(str(acc) + '\n')

        # Loss metrics
        with open(output_dir + '/train_loss.txt', 'w+') as history_file:
            for loss in history.history['loss']:
                history_file.write(str(loss) + '\n')

        with open(output_dir + '/val_loss.txt', 'w+') as history_file:
            for loss in history.history['val_loss']:
                history_file.write(str(loss) + '\n')

    @staticmethod
    def get_versions():
        versions = {
            'python': platform.python_version(),
            'tensorflow': tensorflow.__version__,
            'keras': keras.__version__
        }
        return versions

    def run(self):
        # Read data input
        train_manifest = self.config['data']['input']['train_class_manifest_path']
        if os.path.isfile(train_manifest):
            train_df = pd.read_json(train_manifest)
            self.train_manifest_df = data_utils.check_data_manifest_params(train_df)
        else:
            raise ValueError('train_class_manifest_path must point to a valid file.')

        val_manifest = self.config['data']['input']['val_class_manifest_path']
        if os.path.isfile(val_manifest):
            val_df = pd.read_json(val_manifest)
            self.val_manifest_df = data_utils.check_data_manifest_params(val_df)
        else:
            raise ValueError('val_class_manifest_path must point to a valid file.')

        # Read data_processing values
        self.target_size = self.config['data']['data_processing']['target_size']
        self.batch_size = self.config['data']['data_processing']['batch_size']

        # Load train dataset stats
        if isinstance(self.config['data']['data_processing']['train_stats_pickle'], str):
            train_stats_pickle = self.config['data']['data_processing']['train_stats_pickle']
        else:
            train_stats_pickle = os.path.join(self.output_dir, 'train_stats.pickle')

        if os.path.isfile(train_stats_pickle):
            # Read existing stats
            print('Loading data stats from pickle ', train_stats_pickle)
            with open(train_stats_pickle, 'rb') as handle:
                self.train_data_stats = pickle.load(handle)
        else:
            # Compute data stats
            print('Creating data stats and saving pickle on ', train_stats_pickle)
            self.train_data_stats = data_utils.compute_dataset_statistics(dataframe=self.train_manifest_df,
                                                                          target_size=self.target_size,
                                                                          batch_size=1000,
                                                                          save_name=train_stats_pickle)

        # TODO: save train_statistics as dictionary
        self.config['data']['data_processing'].update({'train_statistics': str(self.train_data_stats)})

        self.model_spec = self.get_model_spec()
        self.config['settings'].update({'model_spec': self.model_spec.as_json()})

        # Set up the training data generator
        train_data_generator = EnhancedImageDataGenerator(
            custom_crop=self.config['data']['train_data_augmentation']['custom_crop'],
            random_crop_size=self.config['data']['train_data_augmentation']['random_crop_size'],
            rotation_range=self.config['data']['train_data_augmentation']['rotation_range'],
            width_shift_range=self.config['data']['train_data_augmentation']['width_shift_range'],
            height_shift_range=self.config['data']['train_data_augmentation']['height_shift_range'],
            preprocessing_function=self.model_spec.preprocess_input,
            shear_range=self.config['data']['train_data_augmentation']['shear_range'],
            zoom_range=self.config['data']['train_data_augmentation']['zoom_range'],
            horizontal_flip=self.config['data']['train_data_augmentation']['horizontal_flip'],
            vertical_flip=self.config['data']['train_data_augmentation']['vertical_flip'],
            fill_mode=self.config['data']['train_data_augmentation']['fill_mode']
        )

        # Set up the validation data generator
        val_data_generator = EnhancedImageDataGenerator(
            custom_crop=self.config['data']['val_data_augmentation']['custom_crop'],
            random_crop_size=self.config['data']['val_data_augmentation']['random_crop_size'],
            rotation_range=self.config['data']['val_data_augmentation']['rotation_range'],
            width_shift_range=self.config['data']['val_data_augmentation']['width_shift_range'],
            height_shift_range=self.config['data']['val_data_augmentation']['height_shift_range'],
            preprocessing_function=self.model_spec.preprocess_input,
            shear_range=self.config['data']['val_data_augmentation']['shear_range'],
            zoom_range=self.config['data']['val_data_augmentation']['zoom_range'],
            horizontal_flip=self.config['data']['val_data_augmentation']['horizontal_flip'],
            vertical_flip=self.config['data']['val_data_augmentation']['vertical_flip'],
            fill_mode=self.config['data']['val_data_augmentation']['fill_mode']
        )

        # Set up the train and validation generator
        self.train_generator = train_data_generator.flow_from_dataframe(
            dataframe=self.train_manifest_df,
            x_col="filename",
            y_col="class_probabilities",
            crop_col="crop",
            class_mode="probabilistic",
            iterator_mode=self.config['data']['data_processing']['iterator_mode'],
            target_size=(self.target_size, self.target_size),
            batch_size=self.batch_size,
            validate_filenames=False
        )

        self.val_generator = val_data_generator.flow_from_dataframe(
            dataframe=self.val_manifest_df,
            x_col="filename",
            y_col="class_probabilities",
            crop_col="crop",
            class_mode="probabilistic",
            iterator_mode=self.config['data']['data_processing']['iterator_mode'],
            target_size=(self.target_size, self.target_size),
            batch_size=self.batch_size,
            validate_filenames=False
        )

        # Load class weights
        class_weights = self.get_class_weights()
        self.config['data']['data_processing'].update({'class_weights_value': class_weights})

        layers_to_freeze = self.freeze_layers()
        self.config['hyperparameters'].update({'layers_to_freeze': layers_to_freeze})

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

        model_output_dir = self.customize_model_output_dir(num_epochs=True,
                                                           lr_sgd=True,
                                                           batch_size=False,
                                                           output_path=self.output_dir)
        utils.mkdir(model_output_dir)

        num_iterations = self.config['settings']['num_iterations']
        track_sensitivity = self.config['settings']['track_sensitivity']
        for iteration in range(0, num_iterations):

            model_dir = os.path.join(model_output_dir, 'iter_' + str(iteration))
            utils.mkdir(model_dir)

            logs_dir = os.path.join(model_dir, 'logs')
            utils.mkdir(logs_dir)

            model = studio.training.keras.Trainer(
                output_model_dir=model_dir,
                output_logs_dir=logs_dir,
                model_spec=self.model_spec,
                # custom_model=custom_model,
                num_classes=len(self.train_data_stats['class_histogram']),
                train_generator=self.train_generator,
                val_generator=self.val_generator,
                batch_size=self.batch_size,
                epochs=self.config['hyperparameters']['num_epochs'],
                freeze_layers_list=self.config['hyperparameters']['layers_to_freeze'],
                loss_function=self.config['hyperparameters']['loss_function'],
                workers=self.config['settings']['num_workers'],
                max_queue_size=self.config['settings']['max_queue_size'],
                num_gpus=self.num_gpus,
                optimizer=self.get_optimizer(),
                class_weights=self.config['data']['data_processing']['class_weights_value'],
                # verbose=True,
                callback_list=self.process_callbacks(),
                track_sensitivity=track_sensitivity,
                dropout_rate=self.config['hyperparameters']['dropout_rate'],
                # That will make the networks invariant to the input size
                input_shape=(None, None, 3),
                save_training_options=False
            )

            # Train the model
            model.run()

            self.save_model_history(model, model_dir)

            if self.num_gpus > 1:
                unparallelize(model_path=os.path.join(model_dir, 'model_max_acc.hdf5'),
                              new_model_name='model_max_acc_1_gpu.hdf5')
                if track_sensitivity:
                    unparallelize(model_path=os.path.join(model_dir, 'model_max_sensitivity.hdf5'),
                                  new_model_name='model_max_sens_1_gpu.hdf5')
                self.model_path = os.path.join(model_dir, 'model_max_acc_1_gpu.hdf5')
            else:
                self.model_path = os.path.join(model_dir, 'model_max_acc.hdf5')

        versions = self.get_versions()
        self.config['history'].update({'versions': versions})
