import pytest

from PIL import Image
from keras_model_specs import ModelSpec
from studio.evaluation.keras.data_generators import EnhancedImageDataGenerator


def test_enhanced_image_data_generator_wrong_scale_size(test_catdog_dataset_path):
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))
    with pytest.raises(ValueError, match='Incorrect format for `scale_sizes`, list of ints or `= default` is expected'):
        test_data_generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                         data_augmentation={'scale_sizes': 'asd'})
        test_data_generator.flow_from_directory(
            directory=test_catdog_dataset_path,
            batch_size=1,
            target_size=model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False)


def test_enhanced_image_data_generator_wrong_transforms(test_catdog_dataset_path):
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))

    with pytest.raises(ValueError, match='Wrong transform: failure . Check documentation to see the supported ones'):
        test_data_generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                         data_augmentation={'scale_sizes': [256],
                                                                            'transforms': ['failure']})
        datagen = test_data_generator.flow_from_directory(
            directory=test_catdog_dataset_path,
            batch_size=1,
            target_size=model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False)
        datagen.next()

    with pytest.raises(ValueError, match='Incorrect format for `transforms`, a list of transforms is expected'):
        test_data_generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                         data_augmentation={'scale_sizes': [256],
                                                                            'transforms': 'blah'})
        datagen = test_data_generator.flow_from_directory(
            directory=test_catdog_dataset_path,
            batch_size=1,
            target_size=model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False)

        datagen.next()


def test_enhanced_image_data_generator_wrong_crop_original(test_catdog_dataset_path):
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))

    with pytest.raises(ValueError,
                       match='crop_original mode entered not supported, only `center_crop` is being supported now'):
        test_data_generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                         data_augmentation={'crop_original': 'fail',
                                                                            'scale_sizes': [256]})
        datagen = test_data_generator.flow_from_directory(
            directory=test_catdog_dataset_path,
            batch_size=1,
            target_size=model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False)

        datagen.next()


def test_enhanced_image_data_generator_wrong_arguments(test_catdog_dataset_path):
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))

    with pytest.raises(ValueError,
                       match='data_augmentation dictionary should contain `crop_original`, `scale_sizes` or '
                             '`transforms` as keys'):
        test_data_generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                         data_augmentation={'error': 123})
        datagen = test_data_generator.flow_from_directory(
            directory=test_catdog_dataset_path,
            batch_size=1,
            target_size=model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False)

        datagen.next()

    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))

    with pytest.raises(ValueError, match='`data_augmentation` is a %s and it should be a dictionary' % type([1, 2, 3])):
        test_data_generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                         data_augmentation=[1, 2, 3])
        datagen = test_data_generator.flow_from_directory(
            directory=test_catdog_dataset_path,
            batch_size=1,
            target_size=model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False)

        datagen.next()


def test_enhanced_image_data_generator(test_catdog_dataset_path, test_catdog_manifest_dataframe):
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))
    test_data_generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                     data_augmentation={'scale_sizes': 'default',
                                                                        'transforms': ['horizontal_flip']})
    # Flow from directory
    datagen = test_data_generator.flow_from_directory(
        directory=test_catdog_dataset_path,
        batch_size=1,
        target_size=model_spec.target_size[:2],
        class_mode='categorical',
        shuffle=False)

    batch_x, batch_y = datagen.next()
    assert batch_x.shape == (1, 144, 224, 224, 3)
    assert len(batch_y) == 144

    # Flow from dataframe
    datagen = test_data_generator.flow_from_dataframe(test_catdog_manifest_dataframe,
                                                      directory=test_catdog_dataset_path,
                                                      batch_size=1,
                                                      x_col="filename",
                                                      y_col="class_probabilities",
                                                      target_size=model_spec.target_size[:2],
                                                      class_mode='probabilistic',
                                                      shuffle=False)

    batch_x, batch_y = datagen.next()
    assert batch_x.shape == (1, 144, 224, 224, 3)
    assert len(batch_y) == 144

    test_data_generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                     data_augmentation={'scale_sizes': [256],
                                                                        'transforms': ['horizontal_flip',
                                                                                       'vertical_flip',
                                                                                       'rotate_90',
                                                                                       'rotate_180',
                                                                                       'rotate_270']})

    datagen = test_data_generator.flow_from_directory(
        directory=test_catdog_dataset_path,
        batch_size=1,
        target_size=model_spec.target_size[:2],
        class_mode='categorical',
        shuffle=False)

    batch_x, batch_y = datagen.next()
    assert batch_x.shape == (1, 108, 224, 224, 3)
    assert len(batch_y) == 108

    # Flow from dataframe
    datagen = test_data_generator.flow_from_dataframe(test_catdog_manifest_dataframe,
                                                      directory=test_catdog_dataset_path,
                                                      batch_size=1,
                                                      x_col="filename",
                                                      y_col="class_probabilities",
                                                      target_size=model_spec.target_size[:2],
                                                      class_mode='probabilistic',
                                                      shuffle=False)
    batch_x, batch_y = datagen.next()
    assert batch_x.shape == (1, 108, 224, 224, 3)
    assert len(batch_y) == 108

    test_data_generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                     data_augmentation={'transforms': ['horizontal_flip',
                                                                                       'vertical_flip',
                                                                                       'rotate_90',
                                                                                       'rotate_180',
                                                                                       'rotate_270']})

    # Flow from directory
    datagen = test_data_generator.flow_from_directory(
        directory=test_catdog_dataset_path,
        batch_size=1,
        target_size=model_spec.target_size[:2],
        class_mode='categorical',
        shuffle=False)

    batch_x, batch_y = datagen.next()
    assert batch_x.shape == (1, 6, 224, 224, 3)
    assert len(batch_y) == 6

    # Flow from dataframe
    datagen = test_data_generator.flow_from_dataframe(test_catdog_manifest_dataframe,
                                                      directory=test_catdog_dataset_path,
                                                      batch_size=1,
                                                      x_col="filename",
                                                      y_col="class_probabilities",
                                                      target_size=model_spec.target_size[:2],
                                                      class_mode='probabilistic',
                                                      shuffle=False)

    batch_x, batch_y = datagen.next()
    assert batch_x.shape == (1, 6, 224, 224, 3)
    assert len(batch_y) == 6

    test_data_generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                     data_augmentation={'scale_sizes': [256],
                                                                        'crop_original': 'center_crop',
                                                                        'transforms': ['horizontal_flip',
                                                                                       'vertical_flip',
                                                                                       'rotate_90',
                                                                                       'rotate_180',
                                                                                       'rotate_270']})
    # Flow from directory
    datagen = test_data_generator.flow_from_directory(
        directory=test_catdog_dataset_path,
        batch_size=1,
        target_size=model_spec.target_size[:2],
        class_mode='categorical',
        shuffle=False)

    batch_x, batch_y = datagen.next()
    assert batch_x.shape == (1, 108, 224, 224, 3)
    assert len(batch_y) == 108

    # Flow from dataframe
    datagen = test_data_generator.flow_from_dataframe(test_catdog_manifest_dataframe,
                                                      directory=test_catdog_dataset_path,
                                                      batch_size=1,
                                                      x_col="filename",
                                                      y_col="class_probabilities",
                                                      target_size=model_spec.target_size[:2],
                                                      class_mode='probabilistic',
                                                      shuffle=False)

    batch_x, batch_y = datagen.next()
    assert batch_x.shape == (1, 108, 224, 224, 3)
    assert len(batch_y) == 108

    # Custom crop with data augmentation
    test_data_generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                     custom_crop=True,
                                                     data_augmentation={'scale_sizes': 'default',
                                                                        'transforms': ['horizontal_flip']})

    datagen = test_data_generator.flow_from_dataframe(test_catdog_manifest_dataframe,
                                                      directory=test_catdog_dataset_path,
                                                      batch_size=1,
                                                      x_col="filename",
                                                      y_col="class_probabilities",
                                                      crop_col="crop",
                                                      target_size=model_spec.target_size[:2],
                                                      class_mode='probabilistic',
                                                      shuffle=False)
    batch_x, batch_y = datagen.next()
    assert batch_x.shape == (1, 144, 224, 224, 3)
    assert len(batch_y) == 144

    # Custom crop without data augmentation
    test_data_generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                     custom_crop=True)

    datagen = test_data_generator.flow_from_dataframe(test_catdog_manifest_dataframe,
                                                      directory=test_catdog_dataset_path,
                                                      batch_size=1,
                                                      x_col="filename",
                                                      y_col="class_probabilities",
                                                      crop_col="crop",
                                                      target_size=model_spec.target_size[:2],
                                                      class_mode='probabilistic',
                                                      shuffle=False)
    batch_x, batch_y = datagen.next()
    # Only image should be generated at a time without augmentation
    assert batch_x.shape == (1, 224, 224, 3)
    assert len(batch_y) == 1


def test_custom_crop(test_catdog_dataset_path, test_catdog_manifest_dataframe, test_single_cat_image_path):
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))
    generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                           custom_crop=True)
    datagen = generator.flow_from_dataframe(test_catdog_manifest_dataframe,
                                            directory=test_catdog_dataset_path,
                                            x_col="filename",
                                            y_col="class_probabilities",
                                            crop_col="crop",
                                            batch_size=1,
                                            target_size=(256, 256))
    x, y = datagen.next()
    assert x.shape == (1, 256, 256, 3)
    assert len(y) == 1
    img = Image.open(test_single_cat_image_path)

    img_cropped = datagen.apply_custom_crop(img, crop_coordinates=[100, 100, 50, 75])
    assert img_cropped.size == (50, 75)

    img_cropped = datagen.apply_custom_crop(img, crop_coordinates=[100, 100, 30, 60])
    assert img_cropped.size == (30, 60)

    img_cropped = datagen.apply_custom_crop(img, crop_coordinates=[100, 100, 100, 120])
    assert img_cropped.size == (100, 120)


def test_custom_crop_fail(test_catdog_dataset_path, test_catdog_manifest_fail_dataframe):
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))
    generator = EnhancedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                           custom_crop=True)
    datagen = generator.flow_from_dataframe(test_catdog_manifest_fail_dataframe,
                                            directory=test_catdog_dataset_path,
                                            x_col="filename",
                                            y_col="class_probabilities",
                                            crop_col="crop",
                                            batch_size=1,
                                            target_size=(256, 256))
    with pytest.raises(TypeError):
        x = datagen.next()[0]
        assert x.shape == (1, 256, 256, 3)

        x = datagen.next()[0]
        assert x.shape == (1, 256, 256, 3)
