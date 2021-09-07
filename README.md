# AI-Studio
[![CircleCI](https://circleci.com/gh/aip-labs/ai-studio.svg?style=shield&circle-token=4576b216fe831d7a6cf2b4e64c651925d166499f)](https://app.circleci.com/pipelines/github/aip-labs/ai-studio)

**AI-Studio** is designed to provide an end-to-end ML system that offers **scalable**, **reproducible**, **reliable**, and **code-free tools** by automating:

1. Data management
2. Model training
3. Model evaluation
4. Experiment visualization
5. Experiment checkpointing


# Commands
To achieve this, AI-Studio is equipped with `commands` that operate with `step` operations with clear objectives and standardized outputs.

## Auto
The `auto` command allows the user to request jobs involving data processing and management, training and evaluation of models.

## Deploy
The deploy command allows the user to perform options related to model deployment such as model conversions.

### Example
- To carry out training, from the `studio` directory, run the script `studio.py` as:
```
python studio.py './examples/7_way_train.yaml'
```

- To convert a Keras model to Tensorflow deployment format, run:
```
python studio.py --deploy ./examples/deployment.yaml

```

# Steps

## Data
The `data` step assumes the user is in possession of the training or evaluation data but the data is not organized in the expected format for the `train` and/or `eval` steps. The processed data is stored in the form of "class_manifest" `json` files that contain "filename" and "class_probabilities" information.

### Train
The `train` data mode loads the training and validation data that will be passed to the `train` step in the following ways.

#### Directory
The `directory` option loads the training and validation data from local directories containing one subdirectory for each training class.

This option expects as inputs:
* `train_dir`: (*string*) path to train folder.
* `val_dir`: (*string*) path to validation folder.

#### Model Map
The `modelmap` option loads the training and validation data as a ModelMap object that is initialized from a "dataset_manifest" and "conditions_manifest" files.

This option expects as inputs:
* `data_directory`: (*string*) path where image files specified in `dataset_manifest_path` are located.
* `dataset_manifest_path`: (*string*) path to `json` manifest that provides information about images, such as "id", "tags", "filename", "storage_key", etc. associated to each image. Normally this manifest is obtained from the AI-Studio Snapshot tab.
* `conditions_manifest_path`: (*string*) path to `json` manifest file that defines the relationship between conditions and diagnosis nodes.
* `validation_split`: portion of data defined in `dataset_manifest_path` that will be used for validation purposes. We currently support:
    * `class_count`: (*integer*) fixed number of validation samples assigned to each class.
    * `class_ratio`: (*float*) validation split fraction assigned to each class.

#### Lab
The `lab` option loads the training and validation data sourced in different ways by the Data-Studio. Currently, the modes supported are:
* `manifest`: this mode reads the train and validation data defined as a Data-Studio's Snapshot manifest with already defined training classes specified in the "dictionary" field. This option expects as inputs:
    * `data_directory`: (*string*) path where image files specified in `train_lab_manifest_path` and `val_lab_manifest_path` are located.
    * `train_lab_manifest_path`: (*string*) Data-Studio Snapshot manifest `json` file with train partition samples.
    * `val_lab_manifest_path`: (*string*) Data-Studio Snapshot manifest `json` file with validation partition samples.

### Evaluation
The `eval` data mode loads the evaluation data that will be passed to the `eval` step in the following ways.

#### Directory
The `directory` option loads the evaluation data from a local directory containing one subdirectory for each test class.

This option expects as input:
* `test_dir`: (*string*) path to test folder.

#### Model Map
The `modelmap` option loads the evaluation data as a ModelMap object that is initialized from a "dataset_manifest" and "conditions_manifest" files.

This option expects as inputs:
* `data_directory`: (*string*) path where image files specified in `dataset_manifest_path` are located.
* `dataset_manifest_path`: (*string*) path to `json` manifest that provides information about images, such as "id", "tags", "filename", "storage_key", etc. associated to each image. Normally this manifest is obtained from the Data-Studio Snapshot tab.
* `conditions_manifest_path`: (*string*) path to `json` manifest file that defines the relationship between conditions and diagnosis nodes.

#### Lab
The `lab` option loads the evaluation data sourced in different ways by the Data-Studio. Currently, the modes supported are:
* `manifest`: this mode reads the evaluation data defined as a Data-Studio's Snapshot manifest with already defined test classes specified in the "dictionary" field. This option expects as inputs:
    * `data_directory`: (*string*) path where image files specified in `test_lab_manifest_path` are located.
    * `test_lab_manifest_path`: (*string*) Lab Snapshot manifest `json` file with test partition samples.


## Train
The `train` step runs a CNN architecture training for a given data, settings, hyperparameters and optimization values using the [`keras-trainer`](https://github.com/aip-labs/keras-trainer) package.

### Data
The `train` data option processes the input data given as a train and validation class manifest `json` files that contain the "filename" and "class_probabilities" information. Note that if the `data` train step has been previously run, the input `train_class_manifest_path` and `val_class_manifest_path` values will be automatically updated.

#### Input
* `train_class_manifest_path`: (*Optional string*) path to `json` manifest with each training image "filename" (full path to image) and "class_probabilities" (array containing float probability values of each class) information. Optionally, it can also include `crop_col` with the list of custom crop coordinates for each image. If unspecified, it will search up for the `data` step output. [[example](https://github.com/aip-labs/ai-studio/blob/master/tests/fixtures/data/animals_train_class_manifest.json)][[source code](https://github.com/aip-labs/keras-trainer/blob/fc4f175537869e57603d7ed37d1c7ec25e9974ab/studio.training.keras/trainer.py#L141-L153)].
* `val_class_manifest_path`: (*Optional string*) path to `json` manifest with each validation image "filename" (full path to image) and "class_probabilities" (array containing float probability values of each class) information. Optionally, it can also include `crop_col` with the list of custom crop coordinates for each image. If unspecified, it will search up for the `data` step output. [[example](https://github.com/aip-labs/ai-studio/blob/master/tests/fixtures/data/animals_val_class_manifest.json)][[source code](https://github.com/aip-labs/keras-trainer/blob/fc4f175537869e57603d7ed37d1c7ec25e9974ab/studio.training.keras/trainer.py#L189-L200)].

#### Data processing
* `target_size`: (*Optional integer*) dimensions to square-resize input images to. If unspecified, `target_size` will default to 224.
* `batch_size`: (*Optional integer*) size of a data batch. If unspecified, `batch_size` will default to 128.
* `preprocess_func`: (*Optional string*) image data processing functions. Currently supported `between_plus_minus_1`, `mean_subtraction`, `bgr_mean_subtraction`, `mean_std_normalization` and `mean_subtraction_plus_minus_1`. If unspecified, `preprocess_func` will default to the one the network was trained on, specified here [[source code](https://github.com/aip-labs/keras-model-specs/blob/54f62f46cd9cd0a2655d5edd3d89be55c0008f56/keras_model_specs/model_spec.py#L50)].
* `subtract_dataset_mean`: (*Optional boolean*) If true will substract the computed dataset mean. Else, will the default `preprocess_args` will be used.
* `class_weights`: (*Optional boolean*) If true, `class_weights_value` will be automatically computed as a dictionary mapping of class indices (integers) to a weight (float) based on the training class representation. These class weight values will be used for weighting the loss function (during training only). If unspecified, `class_weights` will default to True.
* `class_weights_value`: (*Optional list of floats*) containing each train class weight value. These class weight values will be used for weighting the loss function (during training only). If unspecified, `class_weights_value` will take a canonical form.
* `class_weight_smoothing`: (*Optional boolean*) Smoothing value to be applied to `class_weights_value`. [[source code](https://github.com/aip-labs/ai-studio/blob/master/studio/utils/data_utils.py#L142)]. If unspecified, `class_weight_smoothing` will default to 1.0.
* `iterator_mode`: (*Optional string or None*) defines the data sampling mode of the data generator. Currently supported: `None`, `equiprobable`. If `None`, each sample is selected randomly. If `equiprobable`, each sample is selected randomly with uniform class probability, so all the classes are evenly distributed. If unspecified, `iterator_mode` will default to `None`. [[source code](https://github.com/aip-labs/keras-trainer/blob/2a268a1089cc6985983683513efcd5a2cba8e71e/studio.training.keras/data_generators.py#L760-L762)].
* `train_stats_pickle`: (*Optional string*) path to a `pickle` file containing the train dataset stats including `Num_images`, `Mean`, `Std`, and `Class Histogram`. If unspecified, `train_stats_pickle` will default to `None` and these will be automatically computed based on the train dataset manifest. [[source code](https://github.com/aip-labs/ai-studio/blob/18cee72c1cdd148009f219357d5672706252c9c0/studio/utils/data_utils.py#L161)].

#### Data augmentation
This applies to `train_data_augmentation` and `val_data_augmentation` data options.
* `custom_crop`: (*Optional boolean*) if True, custom crops will be applied according to the value in the `crop_col` from the train/validation class manifests. If unspecified, `custom_crop` will default to False. [[source code](https://github.com/aip-labs/keras-trainer/blob/2a268a1089cc6985983683513efcd5a2cba8e71e/studio.training.keras/data_generators.py#L753-L754)].
* `random_crop_size`: (*Optional integer*) Size of the image random crop. Either a percentage of the original image (0,1) that will do square crop, a fixed size (tuple), or integer where the value will set equally to both dimensions. If unspecified, `random_crop_size` will default to 0.0, meaning that any crop will not be applied. [[source code](https://github.com/aip-labs/keras-trainer/blob/2a268a1089cc6985983683513efcd5a2cba8e71e/studio.training.keras/data_generators.py#L750-L752)].
* `rotation_range`: (*Optional integer*) randomly rotate images in the range (degrees, 0 to 180). If unspecified, `rotation_range` will default to 180. [[source code](https://keras.io/preprocessing/image/)]
* `width_shift_range`: (*Optional float*) randomly shift images horizontally (fraction of total width). If unspecified, `width_shift_range` will default to 0.0. [[source code](https://keras.io/preprocessing/image/)]
* `height_shift_range`: (*Optional float*) randomly shift images vertically (fraction of total height). If unspecified, `height_shift_range` will default to 0.0. [[source code](https://keras.io/preprocessing/image/)]
* `shear_range`: (*Optional float*) set range for random image shear. If unspecified, `shear_range` will default to 0.0. [[source code](https://keras.io/preprocessing/image/)]
* `zoom_range`: (*Optional float*) set range for random image zoom. If unspecified, `zoom_range` will default to 0.1. [[source code](https://keras.io/preprocessing/image/)]
* `horizontal_flip`: (*Optional boolean*) if True, randomly flip images respect to the horizontal axis. If unspecified, `horizontal_flip` will default to True. [[source code](https://keras.io/preprocessing/image/)]
* `vertical_flip`: (*Optional boolean*) if True, randomly flip images respect to the vertical axis. If unspecified, `vertical_flip` will default to True. [[source code](https://keras.io/preprocessing/image/)]
* `fill_mode`: (*Optional string*) set mode for filling points outside the input boundaries. If unspecified, `fill_mode` will default to `nearest`. [[source code](https://keras.io/preprocessing/image/)]

#### Settings

* `architecture`: (*string*) Current supported CNN architectures:
    * VGG: `vgg16`, `vgg19`.
    * ResNet: `resnet50`, `resnet101` `resnet152`, `resnet50_v2`, `resnet101_v2`, `resnet152_v2`.
    * ResNeXt: `ResNeXt50`, `ResNeXt101`.
    * MobileNet: `mobilenet_v1`, `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`.
    * Inception: `inception_resnet_v2`, `inception_v3`.
    * Xception: `xception`.
    * NasNet: `nasnet_large`, `nasnet_mobile`.
    * DenseNet: `densenet_169`, `densenet_121`, `densenet_201`.
    * EfficientNet: `efficientnetb0`, `efficientnetb1`, `efficientnetb2`, `efficientnetb3`, `efficientnetb4`, `efficientnetb5`, `efficientnetb6`, `efficientnetb7`.

    The full list of models supported are the ones specified in the [keras-model-specs](https://github.com/aip-labs/keras-model-specs) package.

* `track_sensitivity`: (*Optional boolean*) If True, sensitivity metric will be used as a train callback.
* `num_iterations`: (*Optional integer*) Number of times the training will be ran. If unspecified, `num_iterations` will default to one.
* `num_workers`: (*Optional integer*) Maximum number of processes to spin up when using process-based threading. If unspecified, `num_workers` will default to 1. If 0, will execute the generator on the main thread.
* `max_queue_size`: (*Optional integer*) Maximum size for the generator queue. If unspecified, `max_queue_size` will default to 128.

#### Hyperparameters
* `num_epochs`: (*integer*) Number of epochs to train the model. An epoch is an iteration over the entire training data provided.
* `loss_function`: (*Optional string*) A loss function required to compile the Keras model. If unspecified, `loss_function` will default to `categorical_crossentropy`. [[source code](https://keras.io/losses/)].
* `dropout_rate`: (*Optional float*) Fraction of the input units to drop. If unspecified, `dropout_rate` will default to 0.
* `freeze_layers`: Defines the CNN model layers to freeze. If unspecified, `freeze_layers` will default to `None`.
    * `start`: (*Optional integer*) First layer index to freeze from.
    * `end`: (*Optional integer*) Last layer index to freeze to.
* `cyclical_learning_rate`: Defines a Cyclical Learning Rate (CLR) as a Keras Callback. [[source code](https://github.com/aip-labs/ai-studio/blob/61a9bcd20bdf56e63007469c386fe832c9f8c42b/ai-studio/utils/train_utils.py#L48)].
    * `mode`: (*string*) Currently supported: `triangular`, `triangular2`, `exp_range`.
    * `base_lr`: (*float*) initial learning rate which is the lower boundary in the cycle.
    * `max_lr`:(*float*) upper boundary in the cycle.
    * `step_size`: (*integer*) number of training iterations per half cycle.

### Optimizers
* `SGD`: Defines a Stochastic gradient descent (SGD) optimizer [[source code](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L164)].
    * `lr`: (*Optional float*) Learning rate value. If unspecified, `lr` will default to 0.001.
    * `decay`: (*Optional float*) Decay factor. If unspecified, `decay` will default to 0.0.
    * `momentum`: (*Optional float*) Parameter that accelerates SGD in the relevant direction and dampens oscillations. If unspecified, `momentum` will default to 0.9.
    * `scheduler_frequency`: (*Optional integer list*) list of integers for which the learning rate will decrease based on `gamma` value.
    * `gamma`: (*Optional float*) Gamma value. If unspecified, `gamma` will default to 0.5.
* `Adam`: Defines an Adam optimizer [[source code](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L467)].
    * `lr`: (*Optional float*) Learning rate value. If unspecified, `lr` will default to 0.001.
    * `beta_1`: (*float*), 0 < beta < 1. Generally close to 1, e.g. 0.9.
    * `beta_2`: (*float*), 0 < beta < 1. Generally close to 1, e.g. 0.999.
    * `epsilon`: (*float*) epsilon value.
    * `decay`: (*float*) decay value.


## Eval
The `eval` step runs a CNN model(s) evaluation for a given test data and evaluation values using the evaluator package.

### Data
The `eval` data option processes the input data given as a test class manifest `json` file that contain the "filename" and "class_probabilities" information. Note that if the `data` eval step has been previously run, the input `test_class_manifest_path` value will be automatically updated.

#### Input
* `test_class_manifest_path`: (*Optional string*) path to `json` manifest with each test image "filename" (full path to image) and "class_probabilities" (array containing float probability values of each class) information. Optionally, it can also include `crop_col` with the list of custom crop coordinates for each image. If unspecified, it will search up for the `data` step output. [[example](https://github.com/aip-labs/ai-studio/blob/master/tests/fixtures/data/animals_test_class_manifest.json)].

#### Single
* `model_path`: (*string*) path to a `hdf5` Keras model file.
* `ensemble_models_dir`: (*string*) path to a directory with multiple models organized in single folders where the `hdf5` Keras file and `model_specs.json` live in.
* `combination_mode`: (*Optional string*) Method of combining probability values. Currently supported: `arithmetic`, `geometric`, `maximum`, `harmonic`. If unspecified, `combination_mode` will default to `arithmetic`.
* `concept_dictionary_path`: (*Optional string*) path to a dictionary file containing a mapping between training classes and evaluation classes. If unspecified, `concept_dictionary_path` will default to `None`.
* `top_k`: (*integer*) An integer specifying the k-th highest prediction probabilities to consider, e.g., top_k = 5 is top-5 preds.
* `batch_size`: (*Optional integer*) size of a data batch. If unspecified, `batch_size` will default to 32.
* `custom_crop`: (*Optional boolean*) if True, custom crops will be applied according to the value in the `crop_col` from the train/validation class manifests. If unspecified, `custom_crop` will default to False.
* `data_augmentation`: (*Optional boolean*) Applies data augmentation as defined in the train/data/data processing section. If unspecified, `data_augmentation` will default to `False`.
* `confusion_matrix`: (*boolean*) If True, show the confusion matrix. If unspecified, `confusion_matrix` will default to `False`.
* `show_confusion_matrix_text`: (*boolean*) If True, will plot the confusion matrix as text format. If unspecified, `show_confusion_matrix_text` will default to `False`.
* `verbose`: (*integer*) If True, show evaluator text. If unspecified, `verbose` will default to `False`.

# Visual Classifiers
## Versioning Control
Start using standard versioning MODE-MAJOR.MINOR.PATCH for visual classifiers:
* `MODE`: visual classifier mode, e.g. m (macroscopic), or d (dermoscopic), f (first-gate)
* `MAJOR`: new model map, e.g. updates in ontology, new classes added, etc.
* `MINOR`: update in dataset, e.g. new images added, etc.
* `PATCH`: update in training, e.g. changes in hyper-parameters, architecture, ensemble, etc. 
