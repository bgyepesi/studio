import os
import sys
import math
import time
import pickle
import collections
import numpy as np
import pandas as pd

from studio.utils import utils
from studio.training.keras.data_generators import EnhancedImageDataGenerator


# Solve High Resolution truncated files
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_data_manifest_params(data_df):
    """
    Check `data_df` dataframe contains `filename` and `class_probabilities`
    columns. Converts `class_probabilities` column to list of values.

    Args:
        data_df: (dataframe) Pandas Dataframe with data manifest information.

    Return:
        df: Checked `data_df` with expected format.
    """
    df = data_df.copy()

    # Check dataframe contains `filename` and `class_probabilities` columns
    if not set(['filename', 'class_probabilities']) <= set(df.columns.to_list()):
        raise ValueError('Data manifest must contain `filename` and `class_probabilities` columns.')

    # Check `class_probabilities` column values must be same length
    if not all(df['class_probabilities'].apply(lambda x: len(x) == len(df['class_probabilities'][0]))):
        raise ValueError('All values in `class_probabilities` column must have same length.')

    # Check `class_probabilities` column values' length must be greater than one
    if not all(len(item) > 1 for item in df['class_probabilities']):
        raise ValueError('`class_probabilities` column values must be greater than one.')

    # Convert `class_probabilities` column to list of values
    df['class_probabilities'] = df['class_probabilities'].tolist()
    return df


def directory_to_dataframe(directory):
    """
    Converts the target directory with one subdirectory per class to a Pandas
    Dataframe with `filename` and one-hot encoded `class_probabilities` information.

    Args:
        directory: (string) path to the directory to read images from class subfolders.

    Return:
        directory_df: dataframe containing two columns:
            - 'filename': (string) absolute paths to files.
            - 'class_probabilities': (list) one-hot encoded target data.
    """
    filenames = []
    classes = []
    subfolders_paths = sorted([f.path for f in os.scandir(directory) if f.is_dir()])
    subfolders_names = sorted([os.path.basename(f.path) for f in os.scandir(directory) if f.is_dir()])
    s = pd.Series(subfolders_names)
    one_hot_matrix = pd.get_dummies(s, dtype=float)

    for folder in subfolders_paths:
        files = utils.get_file_paths_recursive(folder)
        class_list = len(files) * [one_hot_matrix[os.path.basename(folder)].to_list()]
        filenames.extend(files)
        classes.extend(class_list)

    data = {'filename': filenames, 'class_probabilities': classes}
    directory_df = pd.DataFrame(data)
    return directory_df


def compute_mean_std(generator, batch_size):
    mean = np.zeros(3)
    std = np.zeros(3)
    mean_aux = np.zeros(3)
    std_aux = np.zeros(3)
    count_batch = 1
    n_total_images = generator.samples
    n_batches = n_total_images // batch_size
    last_imgs = n_total_images % batch_size

    print('Computing mean and std...')
    print('Total Images', n_total_images)
    print('Using %d Batches of size %d ' % (n_batches, batch_size))
    print('Images in last batch ', last_imgs)

    t_t = time.time()
    # High Res. images takes longer
    for x_batch, y_batch in generator:
        # Last Batch
        t = time.time()

        if count_batch > n_batches + 1:
            print('End')
            break

        elif count_batch == n_batches + 1:
            x_batch = x_batch[0:last_imgs]

        for img in x_batch:
            for ch in range(0, 3):
                mean_aux[ch] += np.mean(img[:, :, ch])
                std_aux[ch] += np.std(img[:, :, ch])

        print('Batch number ', count_batch)
        print('Time elapsed', time.time() - t)
        sys.stdout.flush()
        mean += mean_aux / n_total_images
        std += std_aux / n_total_images
        mean_aux = np.zeros(3)
        std_aux = np.zeros(3)
        count_batch += 1

    print('Mean ', mean)
    print('Std ', std)
    print('Total time elapsed ', time.time() - t_t)
    return mean, std


def create_class_histogram(generator):
    # Create histogram of classes
    class_counter = collections.Counter(generator.classes)
    class_histogram = list()
    for k in class_counter.keys():
        print('Class %d has %d images' % (k, class_counter[k]))
        class_histogram.append(class_counter[k])

    # Convert to numpy array
    class_histogram = np.array(class_histogram)

    return class_histogram


def create_class_weights(class_histogram, smooth=False, mu=1.0):
    # Class weight function for unbalanced datasets
    class_weight = dict()
    label = 0
    max_val = np.amax(class_histogram)

    for val in class_histogram:
        if smooth:
            # weight = math.log(mu * max_val / float(val))
            weight = 1 / math.log(mu + (float(val) / max_val))
        else:
            weight = max_val / float(val)

        class_weight[label] = weight if weight > 1.0 else 1.0
        label += 1

    return class_weight


def compute_dataset_statistics(directory=None, dataframe=None, target_size=224, batch_size=500, save_name='',
                               generator=None):
    """"
    Compute Num_images, Mean, Std, Class Histogram from the dataset introduced with the generator.
    Save info in pickle file if you introduce a save_name.
    I've checked that there is no image repetition per batch. In the end all the images are used to compute the stats.

    Args:
        directory: Path to data organized by class folders.
        dataframe: Pandas dataframe pointing to the absolute path to images.
        target_size: Integer indicating the (square) image dimension to be resized.
        batch_size: Size of batches to process.
        save_name: To save a pickle object with the dictionary
        generator: Keras image generator.

    Return:
        A dictionary containing:
            {'num_images': n_total_images,
            'mean': mean,
            'std': std,
            'class_histogram': class_histogram}
    """
    if (directory is None) and (dataframe is None) and (generator is None):
        raise ValueError('At least one of the `directory`, `dataframe` or `generator` should not be None.')

    if directory and dataframe:
        raise ValueError('Either directory or dataframe are supported to compute dataset stats.')

    if generator is None:
        train_datagen = EnhancedImageDataGenerator()

        if directory is not None:
            generator = train_datagen.flow_from_directory(directory,
                                                          target_size=(target_size, target_size),
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          class_mode='probabilistic')
        if dataframe is not None:
            generator = train_datagen.flow_from_dataframe(dataframe,
                                                          x_col="filename",
                                                          y_col="class_probabilities",
                                                          target_size=(target_size, target_size),
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          class_mode='probabilistic',
                                                          validate_filenames=False)

    mean, std = compute_mean_std(generator, batch_size)
    n_total_images = generator.samples
    class_histogram = create_class_histogram(generator)

    dict_stats = {'num_images': n_total_images,
                  'mean': mean,
                  'std': std,
                  'class_histogram': class_histogram}

    if save_name != '':
        # Save Stats in pickle file
        file_pickle = save_name
        with open(file_pickle, 'wb') as handle:
            pickle.dump(dict_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saved pickle file in, ', save_name)

    return dict_stats


def split_dataframe(manifest_df, mode='class_fraction', split_ratio=0.1, split_class_count=None):
    """
    Splits `manifest_df` dataframe for each `mode` into two partitions (`partition_A` and `partition_B`).

    Args:
        manifest_df: Pandas dataframe containing a "class_probabilities" column to use class modes.
        mode: (string) partition mode:
            * "random": randomly partitions the `manifest_df` into `|1-split_ratio|` and `split_ratio`.
            * "class_percentage": samples the `manifest_df` per class into `|1-split_ratio|` and `split_ratio`.
            * "class_count": samples the `manifest_df` per class based on `split_class_threshold`.
        split_ratio: (float) Test split fraction.
        split_class_count: (int) Test number of images per class for `class_number` mode


    Returns:
        partition_A_df: Dataframe partition A representing 1 - `partition_B`.
        partition_B_df: Dataframe partition B that guarantees that union of `partition_A_df` and `partition_B_df` is
        equal to `manifest_df`.
    """
    supported_modes = ['random', 'class_fraction', 'class_count']
    if mode not in supported_modes:
        raise ValueError('Split mode not supported.')

    if not isinstance(split_ratio, float) or split_ratio > 1.0 or split_ratio < 0.0:
        raise ValueError("`split_ratio` value must be a non-negative float smaller or equal than 1.0.")

    if mode == 'random':
        partition_b_df = manifest_df.sample(frac=split_ratio, random_state=1)
    else:
        if 'class_probabilities' in manifest_df.columns.tolist():
            n_labels = len(manifest_df.class_probabilities[0])
            class_labels_list = np.argmax(np.asarray(manifest_df.class_probabilities.tolist()), axis=1)
            if mode == 'class_fraction':
                # class split mode
                frames = [manifest_df[class_labels_list == label].sample(frac=split_ratio, random_state=1)
                          for label in np.arange(n_labels)]
            if mode == 'class_count':
                if split_class_count is None:
                    raise ValueError("`split_class_count` value must be a non-negative integer.")
                frames = [manifest_df[class_labels_list == label].sample(split_class_count,
                                                                         random_state=1) for label in np.arange(n_labels)]
            partition_b_df = pd.concat(frames)
        else:
            print("The column `class_probabilities` should be available a column in the dataframe to use class mode.")

    partition_a_df = manifest_df.drop(partition_b_df.index)

    return partition_a_df, partition_b_df
