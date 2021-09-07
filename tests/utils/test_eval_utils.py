import pandas as pd
import pytest
from studio.utils.eval_utils import (
    categorical_to_sparse,
    lab_eval_report,
)


def test_categorical_to_sparse():
    labels = categorical_to_sparse([[0, 1, 0], [1, 0, 0]])
    assert labels == [1, 0], labels

    with pytest.raises(Exception):
        # Fail on non-binary vectors.
        categorical_to_sparse([[0, 0.9, 0.1], [1, 0, 0]])

    with pytest.raises(Exception):
        # Fail if does not sum to 1.
        categorical_to_sparse([[0, 1, 1], [1, 0, 0]])


def test_lab_eval_report():
    # Map the integer indexes to the class name.
    indexes_to_name = {1: 'class_1', 0: 'class_0'}

    # Three samples with two possible classes.
    images_df = pd.DataFrame({
        'id': ['id_1', 'id_2', 'id_3'],  # ID associated with each image.
        'true_probabilities': [[0, 1], [1, 0], [0, 1]],  # Two possible classes for each image.
        'predicted_probabilities': [[0.17, 0.83], [0.66, 0.34], [0.92, 0.08]],
    })

    lab_report = lab_eval_report(images_df=images_df, indexes_to_name=indexes_to_name, sig_digits=1)

    # Dictionary names should be shown by index order.
    assert lab_report['dictionary_names'] == ['class_0', 'class_1'], lab_report['dictionary_names']

    # Samples in the lab report should match the samples in the DataFrame.
    assert len(lab_report['images']) == len(images_df), len(lab_report['images'])

    sample_index = 1  # Check an individual row.
    # Ensure the sample_index matches the ID.
    assert lab_report['images'][sample_index]['id'] == 'id_2'
    # Check correct name mapping.
    assert lab_report['images'][sample_index]['actual_class_name'] == 'class_0'
    # Check if correct sig digits.
    assert lab_report['images'][sample_index]['predicted_probabilities'] == [0.7, 0.3]
