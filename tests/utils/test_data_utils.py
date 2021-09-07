import os
import pytest
import numpy as np

from studio.utils import data_utils


def test_directory_to_dataframe(animals_catdog_val_path):
    df = data_utils.directory_to_dataframe(animals_catdog_val_path)

    assert df['filename'][0] == os.path.join(animals_catdog_val_path, '00000_cat',
                                             'ilsvrc2012_val_00004612_jpeg-56672.jpg')
    assert df['filename'][5] == os.path.join(animals_catdog_val_path, '00001_dog',
                                             'ilsvrc2012_val_00013007_jpeg-56681.jpg')
    assert df['filename'][10] == os.path.join(animals_catdog_val_path, '00003_turtle',
                                              'ilsvrc2012_val_00002514_jpeg-56636.jpg')

    np.testing.assert_array_equal(df['class_probabilities'][0], [1.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(df['class_probabilities'][5], [0.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(df['class_probabilities'][10], [0.0, 0.0, 0.0, 1.0, 0.0])


def test_split_dataframe_random(snapshot):
    manifest_df = snapshot.manifest_df
    partition_a_df, partition_b_df = data_utils.split_dataframe(manifest_df, split_ratio=0.2, mode='random')
    assert len(partition_a_df) + len(partition_b_df) == len(manifest_df)
    assert not np.any(np.isin(partition_b_df.filename.to_list(), partition_a_df.filename.to_list()))

    partition_a_df, partition_b_df = data_utils.split_dataframe(manifest_df, split_ratio=0.0, mode='random')
    assert len(partition_a_df) == len(manifest_df)
    assert len(partition_b_df.id.to_list()) == 0

    partition_a_df, partition_b_df = data_utils.split_dataframe(manifest_df, split_ratio=1.0, mode='random')
    assert len(partition_b_df) == len(manifest_df)
    assert len(partition_a_df.id.to_list()) == 0

    with pytest.raises(ValueError):
        partition_a_df, partition_b_df = data_utils.split_dataframe(manifest_df, split_ratio=2.0, mode='random')

    with pytest.raises(ValueError):
        partition_a_df, partition_b_df = data_utils.split_dataframe(manifest_df, split_ratio=1.1, mode='random')

    with pytest.raises(ValueError):
        partition_a_df, partition_b_df = data_utils.split_dataframe(manifest_df, split_ratio=-0.1, mode='random')


def test_split_dataframe_class_ratio(animals_snapshot_dataframe):
    split_ratio = 0.2
    # Split 20% based on class fractions
    partition_a_df, partition_b_df = data_utils.split_dataframe(animals_snapshot_dataframe,
                                                                split_ratio=split_ratio, mode='class_fraction')

    assert partition_a_df.shape == (89, 2)
    assert partition_b_df.shape == (21, 2)
    assert len(partition_a_df) + len(partition_b_df) == len(animals_snapshot_dataframe)
    assert not np.any(np.isin(partition_b_df.filename.to_list(), partition_a_df.filename.to_list()))

    partition_a_df, partition_b_df = data_utils.split_dataframe(animals_snapshot_dataframe, split_ratio=0.0)
    assert len(partition_a_df) == len(animals_snapshot_dataframe)
    assert len(partition_b_df.filename.to_list()) == 0

    partition_a_df, partition_b_df = data_utils.split_dataframe(animals_snapshot_dataframe, split_ratio=1.0)
    assert len(partition_b_df) == len(animals_snapshot_dataframe)
    assert not np.any(np.isin(partition_b_df.filename.to_list(), partition_a_df.filename.to_list()))

    with pytest.raises(ValueError):
        partition_a_df, partition_b_df = data_utils.split_dataframe(animals_snapshot_dataframe, split_ratio=2.0)

    with pytest.raises(ValueError):
        partition_a_df, partition_b_df = data_utils.split_dataframe(animals_snapshot_dataframe, split_ratio=1.1)

    with pytest.raises(ValueError):
        partition_a_df, partition_b_df = data_utils.split_dataframe(animals_snapshot_dataframe, split_ratio=-0.1)


def test_split_dataframe_class_count(animals_snapshot_dataframe):
    split_class_count = 1
    n_labels = 5
    partition_a_df, partition_b_df = data_utils.split_dataframe(animals_snapshot_dataframe,
                                                                split_class_count=split_class_count,
                                                                mode='class_count')
    assert len(partition_a_df) + len(partition_b_df) == len(animals_snapshot_dataframe)
    assert len(partition_b_df) == split_class_count * n_labels
    assert partition_a_df.shape == (105, 2)
    assert not np.any(np.isin(partition_b_df.filename.to_list(), partition_a_df.filename.to_list()))

    with pytest.raises(ValueError):
        partition_a_df, partition_b_df = data_utils.split_dataframe(animals_snapshot_dataframe, mode='class_count')

    with pytest.raises(ValueError):
        partition_a_df, partition_b_df = data_utils.split_dataframe(animals_snapshot_dataframe,
                                                                    split_ratio=0.0,
                                                                    mode='class_count')
