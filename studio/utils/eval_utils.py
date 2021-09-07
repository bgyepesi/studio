import numpy as np


def categorical_to_sparse(categorical_samples, raise_errors=True):
    """Convert 2D categorical arrays of one-hot encoded labels to a sparse 1D array of integer labels.

    The conversion is simply an `argmax(.)` over each sample.

    The code checks that the vectors in the `categorical_samples` are binary and sum to 1.

    By default, an error will be thrown if the samples are not categorical. Otherwise, a warning will be printed.

    Args:
        categorical_samples: A 2D list of one-hot encoded labels
            where the largest non-zero index position represents the class label.
        raise_errors: If True, an error is raised, else a warning is printed.

    Returns:
        A list of sparse integer labels for each sample.

    """

    sparse_indexes = []

    # Track errors.
    not_one_hot = []
    not_probability = []

    for row_index, categorical_vector in enumerate(categorical_samples):
        # The conversion from the categorical to sparse labels is done here.
        label_index = np.argmax(categorical_vector)

        if categorical_vector[label_index] != 1:
            # We assume a 1-hot-encoded binary vector.
            # If this is a probabilistic ground-truth label, we likely need to think of a different approach.
            not_one_hot.append(
                {'row_index': row_index, 'label_index': label_index, 'class_probabilities': categorical_vector[label_index]}
            )
            if raise_errors:
                raise ValueError(
                    "Error: `categorical_samples` is not one-hot encoded in row index = {}".format(row_index)
                )

        if not np.isclose(sum(categorical_vector), 1.0):
            # The sum of the vector should be close to 1.
            # If is probabilistic, this could be due to a rounding error.
            # If so, we may want to increase the tolerance of the `isclose()` function.
            not_probability.append(
                {'row_index': row_index, 'sum': sum(categorical_vector), 'class_probabilities': categorical_vector[label_index]}
            )
            if raise_errors:
                raise ValueError(
                    "Error: `categorical_samples` do not sum to 1 in row index = {}".format(row_index)
                )

        sparse_indexes.append(label_index)

    if len(not_one_hot) > 0:
        print("Warning: All `categorical_samples` are not binary encoded one-hot vectors.")
        print(not_one_hot)

    if len(not_probability) > 0:
        print("Warning: All `categorical_samples` did not sum to 1.")
        print(not_probability)

    return sparse_indexes


def lab_eval_report(images_df, indexes_to_name, sig_digits=6, raise_errors=True):
    """Convert the true and predicted probabilities into the expected lab evaluations format.

    Args:
        images_df: DataFrame of the samples and probabilities.
            Expects the columns: `id`, `true_probabilities`, and `predicted_probabilities`.
        indexes_to_name: Dictionary where the keys are integer node indexes,
            and the values are the corresponding string class names.
        sig_digits: Integer representing the number of significant digits to use.
            Rounding errors can introduce discrepancies between the lab evaluations and internal metrics
            if very small probabilities are present.
        raise_errors: If True, will raise an error if the ground truth is not a categorical one-hot binary vector.
            Else, will print a warning to screen.

    Returns:
        A dict mapping keys to the condition names and a list of dicts containing the images' predicted and true labels.
        The keys follow the format as expected by the lab.
        For example,
        {
            'dictionary_names': ['melanoma', 'acne', 'psoriasis'],
            'images': [
                {'id': 3, 'actual_class_name': 'acne', 'predicted_probabilities': [0.1, 0.7, 0.2]},
                {'id': 1, 'actual_class_name': 'melanoma', 'predicted_probabilities': [0.9, 0.1, 0.1]},
            ]
        }
        indicates three possible classes with two images.
    """

    # Copy since we modify the DataFrame in place.
    images_df = images_df.copy()

    # Coverts binary one-hot encoded vectors to integer labels.
    # NOTE: This assumes a single label for each image. Will need to be reworked for probabilistic labels.
    true_labels = categorical_to_sparse(images_df['true_probabilities'], raise_errors=raise_errors)

    # Converts integer labels to the corresponding names.
    images_df['true_label_names'] = [indexes_to_name[true_label] for true_label in true_labels]

    # Format as expected by the lab.
    image_outputs = []
    for _, row in images_df.iterrows():
        image_outputs.append(
            {
                "id": row['id'],  # Image ID.
                "actual_class_name": row["true_label_names"],  # Name of the condition.
                "predicted_probabilities": list(np.round((row["predicted_probabilities"]), sig_digits)),
            }
        )

    # The names of the conditions must be in the same order as the vector of probabilities.
    condition_names = [indexes_to_name[key] for key in sorted(indexes_to_name)]

    # Format as expected by the lab.
    output = {
        'dictionary_names': condition_names,
        'images': image_outputs,
    }

    return output
