import numpy as np
from studio.evaluation.keras.metrics_utils import create_ndcg_inputs, create_unranked_reference_from_probabilities, \
    create_ranked_reference_from_probabilities, create_hypothesis


def test_create_unranked_reference_from_probabilities():
    probabilities = np.array([[0.1, 0.2, 0.3, 0.4], [0.6, 0, 0, 0.4]])
    references_scores, references_indices = create_unranked_reference_from_probabilities(probabilities,
                                                                                         top_y=3,
                                                                                         min_probability=0.05)
    expected_references_scores = np.array([[1, 1, 1], [1, 1, 0]])
    expected_references_indices = np.array(([[3, 2, 1], [0, 3, -1]]))

    np.testing.assert_array_equal(expected_references_scores, references_scores)
    np.testing.assert_array_equal(expected_references_indices, references_indices)

    # min_probability=0.25 will filter the 1st indice of the first sample and put -1 on the third position
    references_scores, references_indices = create_unranked_reference_from_probabilities(probabilities,
                                                                                         top_y=3,
                                                                                         min_probability=0.25)
    expected_references_scores = np.array([[1, 1, 0], [1, 1, 0]])
    expected_references_indices = np.array(([[3, 2, -1], [0, 3, -1]]))

    np.testing.assert_array_equal(expected_references_scores, references_scores)
    np.testing.assert_array_equal(expected_references_indices, references_indices)

    # top_y = 4 and will min_probability=0.25 pad the references to a length of 4
    references_scores, references_indices = create_unranked_reference_from_probabilities(probabilities,
                                                                                         top_y=4,
                                                                                         min_probability=0.25)
    expected_references_scores = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
    expected_references_indices = np.array(([[3, 2, -1, -1], [0, 3, -1, -1]]))

    np.testing.assert_array_equal(expected_references_scores, references_scores)
    np.testing.assert_array_equal(expected_references_indices, references_indices)


def test_create_ranked_reference_from_probabilities():
    probabilities = np.array([[0.1, 0.2, 0.3, 0.4], [0.6, 0, 0, 0.4]])
    references_scores, references_indices = create_ranked_reference_from_probabilities(probabilities,
                                                                                       top_y=3,
                                                                                       min_probability=0.05)
    expected_references_scores = np.array([[0.4444, 0.3333, 0.2222], [0.6, 0.4, 0]])
    expected_references_indices = np.array(([[3, 2, 1], [0, 3, -1]]))

    np.testing.assert_array_almost_equal(expected_references_scores, references_scores, 4)
    np.testing.assert_array_equal(expected_references_indices, references_indices)

    # min_probability=0.25 will filter the 1st indice of the first sample and put -1 on the third position
    references_scores, references_indices = create_ranked_reference_from_probabilities(probabilities,
                                                                                       top_y=3,
                                                                                       min_probability=0.25)
    expected_references_scores = np.array([[0.5714, 0.4285, 0], [0.6, 0.4, 0]])
    expected_references_indices = np.array(([[3, 2, -1], [0, 3, -1]]))

    np.testing.assert_array_almost_equal(expected_references_scores, references_scores, 4)
    np.testing.assert_array_equal(expected_references_indices, references_indices)

    # top_y = 4 and will min_probability=0.25 pad the references to a length of 4
    references_scores, references_indices = create_ranked_reference_from_probabilities(probabilities,
                                                                                       top_y=4,
                                                                                       min_probability=0.25)
    expected_references_scores = np.array([[0.5714, 0.4286, 0, 0], [0.6, 0.4, 0, 0]])
    expected_references_indices = np.array(([[3, 2, -1, -1], [0, 3, -1, -1]]))

    np.testing.assert_array_almost_equal(expected_references_scores, references_scores, 4)
    np.testing.assert_array_equal(expected_references_indices, references_indices)


def test_create_hypothesis():
    references_scores = np.array([[0.7, 0.3, 0, 0], [0.6, 0.4, 0, 0]])
    references_indices = np.array(([[0, 1, -1, -1], [0, 1, -1, -1]]))

    predicted_probabilities = np.array([[0.8, 0.1, 0.01, 0.09], [0.0, 0.0, 1.0, 0.0]])
    hypothesis = create_hypothesis(predicted_probabilities, references_indices, references_scores, top_x=4)

    # First sample has 100% match, second sample 0% match
    expected_hypothesis = np.array([[0.7, 0.3, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(expected_hypothesis, hypothesis)

    predicted_probabilities = np.array([[0.01, 0.1, 0.09, 0.9], [0, 1, 0, 0]])
    hypothesis = create_hypothesis(predicted_probabilities, references_indices, references_scores, top_x=4)

    # Predicted CNN indices sorted by probability: [3, 1, 2, 0]
    # By looking at the reference indices 0 and 1 are there in positions 1 and 3 so the scores are assigned on that
    # positions. On the second sample  2st indice is top probability and second indice in reference indices, so the
    # hypothesis will have 0.4 in the first position
    expected_hypothesis = np.array([[0.0, 0.3, 0.0, 0.7], [0.4, 0.0, 0, 0]])
    np.testing.assert_array_equal(expected_hypothesis, hypothesis)


def test_create_ndcg_inputs():
    predicted_probabilities = np.array([[0.8, 0.1, 0.01, 0.09], [0.0, 0.0, 1.0, 0.0]])
    ground_truth_probabilities = np.array([[0.7, 0.3, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0]])

    references_scores, references_indices, hypothesis = create_ndcg_inputs(predicted_probabilities,
                                                                           ground_truth_probabilities,
                                                                           ranked=True,
                                                                           top_x=4,
                                                                           top_y=4
                                                                           )
    expected_references_scores = np.array([[0.7, 0.3, 0, 0], [0.6, 0.4, 0, 0]])
    expected_references_indices = np.array(([[0, 1, -1, -1], [0, 1, -1, -1]]))
    expected_hypothesis = np.array([[0.7, 0.3, 0, 0], [0, 0, 0, 0]])

    np.testing.assert_array_almost_equal(expected_references_scores, references_scores, 4)
    np.testing.assert_array_almost_equal(expected_hypothesis, hypothesis, 4)
    np.testing.assert_array_equal(expected_references_indices, references_indices)
