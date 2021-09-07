import os
import PIL
import math
import warnings
import numpy as np
import multiprocessing.pool


from PIL import Image
from keras_preprocessing import get_keras_submodule
from keras_preprocessing.image.iterator import BatchFromFilesMixin, Iterator
from keras_preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array
from keras_preprocessing.image.utils import validate_filename, _list_valid_filenames_in_directory


backend = get_keras_submodule('backend')

if Image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(Image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = Image.HAMMING
    if hasattr(Image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = Image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(Image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = Image.LANCZOS


class EnhancedBatchFromFilesMixin(BatchFromFilesMixin):
    """Adds methods related to getting batches from filenames.
    It includes the logic to transform image files to batches.
    Addition of a method to random crop images.
    '''

    EnhancedBatchFromFilesMixin inherits from BatchFromFilesMixin:
    (https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/iterator.py#L130)

    This implementation adds the functionality of computing multiple crops following the work Going Deeper with
    Convolutions (https://arxiv.org/pdf/1409.4842.pdf) and allowing the use of transforms on such crops.
    It includes the addition of data_augmentation as an argument. It is a dictionary consisting of 3 elements:

    - 'scale_sizes': 'default' (4 similar scales to Original paper) or a list of sizes. Each scaled image then
    will be cropped into three square parts. For each square, we then take the 4 corners and the center "target_size"
    crop as well as the square resized to "target_size".
    - 'transforms': list of transforms to apply to these crops in addition to not
    applying any transform ('horizontal_flip', 'vertical_flip', 'rotate_90', 'rotate_180', 'rotate_270' are
    supported now).
    - 'crop_original': 'center_crop' mode allows to center crop the original image prior do the rest of transforms,
    scalings + croppings.

    If 'scale_sizes' is None the image will be resized to "target_size" and transforms will be applied over that image.

    For instance: data_augmentation={'scale_sizes':'default', 'transforms':['horizontal_flip', 'rotate_180'],
    'crop_original':'center_crop'}

    For 144 crops as GoogleNet paper, select data_augmentation={'scale_sizes':'default',
    'transforms':['horizontal_flip']}
    This results in 4x3x6x2 = 144 crops per image.

    '''
    """

    def set_processing_attrs(self,
                             image_data_generator,
                             custom_crop,
                             data_augmentation,
                             target_size,
                             color_mode,
                             data_format,
                             save_to_dir,
                             save_prefix,
                             save_format,
                             subset,
                             interpolation):
        """Sets attributes to use later for processing files into a batch.
        # Arguments
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization.
            custom_crop: If True, will crop images according to the dataframe's crop coordinates information contained in
            `crop_col`. The custom crop will be performed before the data_augmentation if both are True.
            data_augmentation : Explained above
            target_size: tuple of integers, dimensions to resize input images to.
            color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
                Color mode to read images.
            data_format: String, one of `channels_first`, `channels_last`.
            save_to_dir: Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample
                images (if `save_to_dir` is set).
            save_format: Format to use for saving sample images
                (if `save_to_dir` is set).
            subset: Subset of data (`"training"` or `"validation"`) if
                validation_split is set in ImageDataGenerator.
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic".
                If PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
        """

        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.data_augmentation = data_augmentation

        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')

        self.color_mode = color_mode
        self.data_format = data_format
        self.custom_crop = custom_crop

        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        if self.interpolation not in _PIL_INTERPOLATION_METHODS:
            raise ValueError(
                'Invalid interpolation method {} specified. Supported '
                'methods are {}'.format(
                    self.interpolation,
                    ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
        self.resample = _PIL_INTERPOLATION_METHODS[self.interpolation]

        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError(
                    'Invalid subset name: %s;'
                    'expected "training" or "validation"' % (subset,))
        else:
            split = None
        self.split = split
        self.subset = subset

        if self.data_augmentation is not None:
            self._check_data_augmentation_keys(data_augmentation)
            self.crop_original = None
            if 'crop_original' in data_augmentation.keys():
                self.crop_original = self.data_augmentation['crop_original']

            self.transforms = ['none']
            if 'transforms' in self.data_augmentation.keys():
                if isinstance(self.data_augmentation['transforms'], list):
                    self.transforms += self.data_augmentation['transforms']
                else:
                    raise ValueError('Incorrect format for `transforms`, a list of transforms is expected')
            self.n_transforms = len(self.transforms)

            if 'scale_sizes' in self.data_augmentation.keys():
                if self.data_augmentation['scale_sizes'] == 'default':
                    self.scale_sizes = self._get_default_sizes(self.target_size[0])
                elif isinstance(self.data_augmentation['scale_sizes'], list) and \
                        all(isinstance(x, int) for x in self.data_augmentation['scale_sizes']):
                    self.scale_sizes = []
                    for size in self.data_augmentation['scale_sizes']:
                        size = round(size)
                        # Use sizes that are multiples of 2
                        if size % 2 != 0:
                            size += 1
                        self.scale_sizes.append(size)
                else:
                    raise ValueError('Incorrect format for `scale_sizes`, list of ints or `= default` is expected')

                self.n_crops = len(self.scale_sizes) * 6 * 3 * self.n_transforms
            else:
                self.scale_sizes = None
                self.n_crops = self.n_transforms

    @staticmethod
    def apply_custom_crop(image, crop_coordinates):
        """
        The format of the image `crop_coordinates` should be [x, y, width, height] and PIL cropping format is
        [left, upper, right, lower].
        """
        crop_x, crop_y, crop_w, crop_h = crop_coordinates

        left = crop_x
        upper = crop_y
        right = crop_x + crop_w
        lower = crop_y + crop_h

        coordinates = [left, upper, right, lower]

        return image.crop(coordinates)

    @staticmethod
    def _check_data_augmentation_keys(data_augmentation):
        if isinstance(data_augmentation, dict):
            keys = data_augmentation.keys()
            if 'scale_sizes' not in keys and 'transforms' not in keys and 'crop_original' not in keys:
                raise ValueError('data_augmentation dictionary should contain '
                                 '`crop_original`, `scale_sizes` or `transforms` as keys')
        else:
            raise ValueError('`data_augmentation` is a %s and it should be a dictionary' % type(data_augmentation))

    @staticmethod
    def _get_default_sizes(target_size, multipliers=(1.1, 1.2, 1.3, 1.4)):
        sizes = []
        for multiplier in multipliers:
            size = round(target_size * multiplier)
            if size % 2 != 0:
                size += 1
            sizes.append(size)
        return sizes

    @staticmethod
    def _get_3_crops(image):
        """

        Args:
            image: PIL Image

        Returns: 3 square sized crops of the image. Top, central and bottom in the case of a vertical image
        and left, central and right in the case of a horizontal image.

        """

        w, h = image.size
        w_center = w / 2
        h_center = h / 2

        if w >= h:
            im_1 = image.crop((0, 0, h, h))
            im_2 = image.crop((w_center - h / 2, 0, w_center + h / 2, h))
            im_3 = image.crop((w - h, 0, w, h))
        else:
            im_1 = image.crop((0, 0, w, w))
            im_2 = image.crop((0, h_center - w / 2, w, h_center + w / 2))
            im_3 = image.crop((0, h - w, w, h))

        return [im_1, im_2, im_3]

    @staticmethod
    def _apply_transform(image, transform):
        '''

        Args:
            image: PIL input image
            transform: Transform to apply

        Returns: Transformed image in PIL format.

        '''
        transform_dict = {'horizontal_flip': PIL.Image.FLIP_LEFT_RIGHT, 'vertical_flip': PIL.Image.FLIP_TOP_BOTTOM,
                          'rotate_90': PIL.Image.ROTATE_90, 'rotate_180': PIL.Image.ROTATE_180,
                          'rotate_270': PIL.Image.ROTATE_270}
        if transform == 'none':
            return image
        elif transform in transform_dict.keys():
            return image.transpose(transform_dict[transform])
        else:
            raise ValueError('Wrong transform: %s . Check documentation to see the supported ones' % transform)

    def _apply_augmentation(self, image, size, transforms):
        '''

        Args:
            image: PIL input image
            size: Target output size
            transforms: List of transforms to apply

        Returns: Crops plus transformations done to the input image

        '''
        crops = []

        target_w, target_h = size
        images_cropped_at_scale = self._get_3_crops(image)

        for img in images_cropped_at_scale:
            w, h = img.size
            w_center = w / 2
            h_center = h / 2

            for transform in transforms:
                # Central Crop
                crops.append(self._apply_transform(img.crop((w_center - target_w / 2,
                                                             h_center - target_h / 2,
                                                             w_center + target_w / 2,
                                                             h_center + target_h / 2))
                                                   .resize((target_w, target_h), resample=self.resample), transform))
                # Left-Up
                crops.append(self._apply_transform(img.crop((0, 0, target_w, target_h)), transform))
                # Left-Bottom
                crops.append(self._apply_transform(img.crop((0, h - target_h, target_w, h)), transform))
                # Right-Up
                crops.append(self._apply_transform(img.crop((w - target_w, 0, w, target_h)), transform))
                # Right-Bottom
                crops.append(self._apply_transform(img.crop((w - target_w, h - target_h, w, h)), transform))
                # Resized Square
                crops.append(self._apply_transform(img.resize((target_w, target_h), resample=self.resample), transform))

        return crops

    def _get_batches_of_transformed_samples(self, index_array):
        grayscale = self.color_mode == 'grayscale'

        if self.data_augmentation is not None:
            batch_x = np.zeros((len(index_array), self.n_crops,) + self.image_shape, dtype=backend.floatx())
        else:
            batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        # Build batch of image data
        for i, j in enumerate(index_array):
            crops = []
            fname = self.filenames[j]
            image = load_img(os.path.join(self.directory, fname),
                             grayscale=grayscale,
                             target_size=None,
                             interpolation=self.interpolation)
            if self.custom_crop:  # expects only a list of integers or missing values
                if isinstance(self.crops[j], list):
                    image = self.apply_custom_crop(image, self.crops[j])
                elif self.crops[j] is None:
                    pass
                # for nan values and throws an error if the input is not float e.g. string
                elif math.isnan(self.crops[j]):
                    pass

            if self.data_augmentation is not None:
                if self.crop_original == 'center_crop':
                    w, h = image.size
                    if w > h:
                        image = image.crop((w / 2 - h / 2, 0, w / 2 + h / 2, h))
                    else:
                        image = image.crop((0, h / 2 - w / 2, w, h / 2 + w / 2))
                elif self.crop_original:
                    raise ValueError('crop_original mode entered not supported, only `center_crop` is being supported now')

                image_w, image_h = image.size

                if self.scale_sizes is not None:
                    for size in self.scale_sizes:
                        if image_w <= image_h:
                            img = image.resize((size, round(image_h / image_w * size)), resample=self.resample)
                        else:
                            img = image.resize((round(image_w / image_h * size), size), resample=self.resample)
                        crops += self._apply_augmentation(img, self.target_size, self.transforms)
                else:
                    crops += [self._apply_transform(image.resize(self.target_size, resample=self.resample), transform)
                              for transform in self.transforms]

                for c_i, crop in enumerate(crops):
                    x = img_to_array(crop, data_format=self.data_format)
                    x = self.image_data_generator.standardize(x)
                    batch_x[i, c_i] = x
            else:
                if len(self.target_size) == 2:
                    image = image.resize(self.target_size, resample=self.resample)
                x = img_to_array(image, data_format=self.data_format)
                # Pillow images should be closed after `load_img`,
                # but not PIL images.
                if hasattr(image, 'close'):
                    image.close()

                if self.image_data_generator:
                    params = self.image_data_generator.get_random_transform(x.shape)
                    x = self.image_data_generator.apply_transform(x, params)
                    x = self.image_data_generator.standardize(x)
                batch_x[i] = x

        # Optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                for crop in batch_x[i]:
                    img = array_to_img(crop, self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e7),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        # Build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(backend.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=backend.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif self.class_mode == 'raw' or self.class_mode == 'probabilistic':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.data_augmentation is not None:
            return batch_x, [batch_y for n in range(self.n_crops)]
        else:
            return batch_x, batch_y


class EnhancedDataFrameIterator(EnhancedBatchFromFilesMixin, Iterator):
    """Iterator capable of reading images from a directory on disk through a dataframe.
        It includes the addition of `data_augmentation` as an argument. It is a dictionary consisting of 3 elements:

        - 'scale_sizes': 'default' (4 similar scales to Original paper) or a list of sizes. Each scaled image then
        will be cropped into three square parts. For each square, we then take the 4 corners and the center "target_size"
        crop as well as the square resized to "target_size".
        - 'transforms': list of transforms to apply to these crops in addition to not
        applying any transform ('horizontal_flip', 'vertical_flip', 'rotate_90', 'rotate_180', 'rotate_270' are
        supported now).
        - 'crop_original': 'center_crop' mode allows to center crop the original image prior do the rest of transforms,
        scalings + croppings.

        If 'scale_sizes' is None the image will be resized to "target_size" and transforms will be applied over that image.

        For instance: data_augmentation={'scale_sizes':'default', 'transforms':['horizontal_flip', 'rotate_180'],
        'crop_original':'center_crop'}

        For 144 crops as GoogleNet paper, select data_augmentation={'scale_sizes':'default',
        'transforms':['horizontal_flip']}
        This results in 4x3x6x2 = 144 crops per image.

    Args:
        dataframe: Pandas dataframe containing the filepaths relative to
            `directory` (or absolute paths if `directory` is None) of the
            images in a string column. It should include other column/s
            depending on the `class_mode`:
            - if `class_mode` is `"categorical"` (default value) it must
                include the `y_col` column with the class/es of each image.
                Values in column can be string/list/tuple if a single class
                or list/tuple if multiple classes.
            - if `class_mode` is `"binary"` or `"sparse"` it must include
                the given `y_col` column with class values as strings.
            - if `class_mode` is `"raw"` or `"multi_output"` it should contain
                the columns specified in `y_col`.
            - if `class_mode` is `"input"` or `None` no extra column is needed.
        directory: string, path to the directory to read images from. If `None`,
            data in `x_col` column should be absolute paths.
        image_data_generator: Instance of `ImageDataGenerator` to use for
            random transformations and normalization. If None, no transformations
            and normalizations are made.
        n_outputs: Integer indicating the number of outputs of the model. It will duplicate the labels. That is useful for
             multi-loss functions.
        iterator_mode: - None: Each sample is selected randomly.
                       - 'equiprobable': Each sample is selected randomly with uniform class probability, so all the
                     classes are evenly distributed. We can have repetition of samples during the same epoch.
        x_col: string, column in `dataframe` that contains the filenames (or
            absolute paths if `directory` is `None`).
        y_col: string or list, column/s in `dataframe` that has the target data.
        crop_col: list, (optional) column in `dataframe` that contains the custom crop coordinates.
        weight_col: string, column in `dataframe` that contains the sample
            weights. Default: `None`.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        classes: Optional list of strings, classes to use (e.g. `["dogs", "cats"]`).
            If None, all classes in `y_col` will be used.
        class_mode: one of "binary", "categorical", "input", "multi_output",
            "raw", "sparse" or None. Default: "categorical".
            Mode for yielding the targets:
            - `"binary"`: 1D numpy array of binary labels,
            - `"categorical"`: 2D numpy array of one-hot encoded labels.
                Supports multi-label output.
            - `"input"`: images identical to input images (mainly used to
                work with autoencoders),
            - `"multi_output"`: list with the values of the different columns,
            - `"raw"`: numpy array of values in `y_col` column(s),
            - `"sparse"`: 1D numpy array of integer labels,
            - `"probabilistic"`: List of probabilities
            - `None`, no targets are returned (the generator will only yield
                batches of image data, which is useful to use in
                `model.predict_generator()`).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        dtype: Dtype to use for the generated arrays.
        validate_filenames: Boolean, whether to validate image filenames in
        `x_col`. If `True`, invalid images will be ignored. Disabling this option
        can lead to speed-up in the instantiation of this class. Default: `True`.
    """
    allowed_class_modes = {
        'binary', 'categorical', 'input', 'multi_output', 'raw', 'sparse', None, 'probabilistic'
    }

    def __init__(self,
                 dataframe,
                 custom_crop=True,
                 data_augmentation=None,
                 image_data_generator=None,
                 directory=None,
                 x_col="filename",
                 y_col="class",
                 crop_col="crop",
                 weight_col=None,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='probabilistic',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 validate_filenames=False):

        super(EnhancedDataFrameIterator, self).set_processing_attrs(image_data_generator,
                                                                    custom_crop,
                                                                    data_augmentation,
                                                                    target_size,
                                                                    color_mode,
                                                                    data_format,
                                                                    save_to_dir,
                                                                    save_prefix,
                                                                    save_format,
                                                                    subset,
                                                                    interpolation)
        df = dataframe.copy()
        self.directory = directory or ''
        self.class_mode = class_mode
        self.dtype = dtype

        if data_augmentation is not None:
            batch_size = 1
        # check that inputs match the required class_mode
        self._check_params(df, x_col, y_col, weight_col, classes)
        if validate_filenames:  # check which image files are valid and keep them
            df = self._filter_valid_filepaths(df, x_col)

        if class_mode == "probabilistic":
            # Probabilistic target array
            self._targets = np.array([value for value in df[y_col].values])
            # We assign only the class with more probabilty
            self.classes = np.array([np.argmax(target) for target in self._targets])
            num_classes = len(self._targets[0])
        if class_mode not in ["input", "multi_output", "raw", "probabilistic", None]:
            df, classes = self._filter_classes(df, y_col, classes)
            num_classes = len(classes)
            # build an index of all the unique classes
            self.class_indices = dict(zip(classes, range(len(classes))))
        # retrieve only training or validation set
        if self.split:
            num_files = len(df)
            start = int(self.split[0] * num_files)
            stop = int(self.split[1] * num_files)
            df = df.iloc[start: stop, :]
        # get labels for each observation
        if class_mode not in ["input", "multi_output", "raw", "probabilistic", None]:
            self.classes = self.get_classes(df, y_col)
        self.filenames = df[x_col].tolist()
        if custom_crop:
            if crop_col in df.columns:
                self.crops = df[crop_col].tolist()
            else:
                raise ValueError('Custom crop selected but %s not found in df columns.' % crop_col)

        self._sample_weight = df[weight_col].values if weight_col else None

        if class_mode == "multi_output":
            self._targets = [np.array(df[col].tolist()) for col in y_col]
        if class_mode == "raw":
            self._targets = df[y_col].values
        self.samples = len(self.filenames)
        validated_string = 'validated' if validate_filenames else 'non-validated'
        if class_mode in ["input", "multi_output", "raw", None]:
            print('Found {} {} image filenames.'
                  .format(self.samples, validated_string))
        else:
            print('Found {} {} image filenames belonging to {} classes.'
                  .format(self.samples, validated_string, num_classes))
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        # Init Iterator
        super(EnhancedDataFrameIterator, self).__init__(self.samples,
                                                        batch_size,
                                                        shuffle,
                                                        seed)
        self.num_classes = num_classes

    def _check_params(self, df, x_col, y_col, weight_col, classes):
        # check class mode is one of the currently supported
        if self.class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(self.class_mode, self.allowed_class_modes))

        if self.class_mode == 'probabilistic':
            if not all(df[y_col].apply(lambda x: isinstance(x, list))):
                raise TypeError('All values in column y_col={} must be list.'
                                .format(x_col))
            class_number = len(df[y_col].values[0])
            if not all(df[y_col].apply(lambda x: len(x) == class_number)):
                raise TypeError('All lists in column y_col={} must have same length.'
                                .format(x_col))

        # check that y_col has several column names if class_mode is multi_output
        if (self.class_mode == 'multi_output') and not isinstance(y_col, list):
            raise TypeError(
                'If class_mode="{}", y_col must be a list. Received {}.'
                .format(self.class_mode, type(y_col).__name__)
            )
        # check that filenames/filepaths column values are all strings
        if not all(df[x_col].apply(lambda x: isinstance(x, str))):
            raise TypeError('All values in column x_col={} must be strings.'
                            .format(x_col))
        # check labels are string if class_mode is binary or sparse
        if self.class_mode in {'binary', 'sparse'}:
            if not all(df[y_col].apply(lambda x: isinstance(x, str))):
                raise TypeError('If class_mode="{}", y_col="{}" column '
                                'values must be strings.'
                                .format(self.class_mode, y_col))
        # check that if binary there are only 2 different classes
        if self.class_mode == 'binary':
            if classes:
                classes = set(classes)
                if len(classes) != 2:
                    raise ValueError('If class_mode="binary" there must be 2 '
                                     'classes. {} class/es were given.'
                                     .format(len(classes)))
            elif df[y_col].nunique() != 2:
                raise ValueError('If class_mode="binary" there must be 2 classes. '
                                 'Found {} classes.'.format(df[y_col].nunique()))
        # check values are string, list or tuple if class_mode is categorical
        if self.class_mode == 'categorical':
            types = (str, list, tuple)
            if not all(df[y_col].apply(lambda x: isinstance(x, types))):
                raise TypeError('If class_mode="{}", y_col="{}" column '
                                'values must be type string, list or tuple.'
                                .format(self.class_mode, y_col))
        # raise warning if classes are given but will be unused
        if classes and self.class_mode in {"input", "multi_output", "raw", None}:
            warnings.warn('`classes` will be ignored given the class_mode="{}"'
                          .format(self.class_mode))
        # check that if weight column that the values are numerical
        if weight_col and not issubclass(df[weight_col].dtype.type, np.number):
            raise TypeError('Column weight_col={} must be numeric.'
                            .format(weight_col))

    def get_classes(self, df, y_col):
        labels = []
        for label in df[y_col]:
            if isinstance(label, (list, tuple)):
                labels.append([self.class_indices[lbl] for lbl in label])
            else:
                labels.append(self.class_indices[label])
        return labels

    @staticmethod
    def _filter_classes(df, y_col, classes):
        df = df.copy()

        def remove_classes(labels, classes):
            if isinstance(labels, (list, tuple)):
                labels = [cls for cls in labels if cls in classes]
                return labels or None
            elif isinstance(labels, str):
                return labels if labels in classes else None
            else:
                raise TypeError(
                    "Expect string, list or tuple but found {} in {} column "
                    .format(type(labels), y_col)
                )

        if classes:
            classes = set(classes)  # sort and prepare for membership lookup
            df[y_col] = df[y_col].apply(lambda x: remove_classes(x, classes))
        else:
            classes = set()
            for v in df[y_col]:
                if isinstance(v, (list, tuple)):
                    classes.update(v)
                else:
                    classes.add(v)
        return df.dropna(subset=[y_col]), sorted(classes)

    def _filter_valid_filepaths(self, df, x_col):
        """Keep only dataframe rows with valid filenames
        Args:
            df: Pandas dataframe containing filenames in a column
            x_col: string, column in `df` that contains the filenames or filepaths
       Returns:
            absolute paths to image files
        """
        filepaths = df[x_col].map(
            lambda fname: os.path.join(self.directory, fname)
        )
        mask = filepaths.apply(validate_filename, args=(self.white_list_formats,))
        n_invalid = (~mask).sum()
        if n_invalid:
            warnings.warn(
                'Found {} invalid image filename(s) in x_col="{}". '
                'These filename(s) will be ignored.'
                .format(n_invalid, x_col)
            )
        return df[mask]

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        if self.class_mode in {"multi_output", "raw", "probabilistic"}:
            return self._targets
        else:
            return self.classes

    @property
    def sample_weight(self):
        return self._sample_weight


class EnhancedDirectoryIterator(EnhancedBatchFromFilesMixin, Iterator):
    """

    It includes the addition of data_augmentation as an argument. It is a dictionary consisting of 3 elements:

    - 'scale_sizes': 'default' (4 similar scales to Original paper) or a list of sizes. Each scaled image then
    will be cropped into three square parts. For each square, we then take the 4 corners and the center "target_size"
    crop as well as the square resized to "target_size".
    - 'transforms': list of transforms to apply to these crops in addition to not
    applying any transform ('horizontal_flip', 'vertical_flip', 'rotate_90', 'rotate_180', 'rotate_270' are
    supported now).
    - 'crop_original': 'center_crop' mode allows to center crop the original image prior do the rest of transforms,
    scalings + croppings.

    If 'scale_sizes' is None the image will be resized to "target_size" and transforms will be applied over that image.

    For instance: data_augmentation={'scale_sizes':'default', 'transforms':['horizontal_flip', 'rotate_180'],
    'crop_original':'center_crop'}

    For 144 crops as GoogleNet paper, select data_augmentation={'scale_sizes':'default',
    'transforms':['horizontal_flip']}
    This results in 4x3x6x2 = 144 crops per image.

    """

    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

    def __init__(self, directory, image_data_generator,
                 custom_crop=False,
                 data_augmentation=None,
                 target_size=(299, 299),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'
                 ):

        super(EnhancedDirectoryIterator, self).set_processing_attrs(image_data_generator,
                                                                    custom_crop,
                                                                    data_augmentation,
                                                                    target_size,
                                                                    color_mode,
                                                                    data_format,
                                                                    save_to_dir,
                                                                    save_prefix,
                                                                    save_format,
                                                                    subset,
                                                                    interpolation)
        self.data_augmentation = data_augmentation
        if self.data_augmentation is not None:
            batch_size = 1
        self.directory = directory
        self.classes = classes

        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()

        # Second, build an index of the images in the different class subfolders.
        results = []
        self.filenames = []
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                pool.apply_async(_list_valid_filenames_in_directory,
                                 (dirpath, self.white_list_formats, self.split,
                                  self.class_indices, follow_links)))
        classes_list = []
        for res in results:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames += filenames
        self.samples = len(self.filenames)
        self.classes = np.zeros((self.samples,), dtype='int32')
        for classes in classes_list:
            self.classes[i:i + len(classes)] = classes
            i += len(classes)

        print('Found %d images belonging to %d classes.' %
              (self.samples, self.num_classes))
        pool.close()
        pool.join()
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        # Init iterator
        super(EnhancedDirectoryIterator, self).__init__(self.samples,
                                                        batch_size,
                                                        shuffle,
                                                        seed)

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return self.classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None


class EnhancedImageDataGenerator(ImageDataGenerator):
    """

    This is a modification of the Keras class ImageDataGenerator that includes some extra functionalities.
    It includes the addition of data_augmentation as an argument for `flow_from_directory and `flow_from_dataframe`.

    The `flow_from_dataframe` function accepts class_mode='probabilistic' to handle a list of labels in a
    dataframe column. It also excepts `custom_crop` as an argument which performs custom crops on images.

    """

    def __init__(self,
                 custom_crop=False,
                 data_augmentation=None,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format='channels_last',
                 validation_split=0.0,
                 ):

        super(EnhancedImageDataGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            validation_split=validation_split)

        self.custom_crop = custom_crop
        self.data_augmentation = data_augmentation

    def flow_from_dataframe(self,
                            dataframe,
                            directory=None,
                            x_col="filename",
                            y_col="class",
                            crop_col="crop",
                            weight_col=None,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='probabilistic',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            data_format='channels_last',
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            subset=None,
                            interpolation='nearest',
                            dtype='float32',
                            validate_filenames=False
                            ):

        return EnhancedDataFrameIterator(
            dataframe,
            custom_crop=self.custom_crop,
            data_augmentation=self.data_augmentation,
            directory=directory,
            image_data_generator=self,
            x_col=x_col,
            y_col=y_col,
            crop_col=crop_col,
            weight_col=weight_col,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            interpolation=interpolation,
            dtype=dtype,
            validate_filenames=validate_filenames
        )

    def flow_from_directory(self,
                            directory,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):

        return EnhancedDirectoryIterator(
            directory=directory,
            image_data_generator=self,
            data_augmentation=self.data_augmentation,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)
