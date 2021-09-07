import numpy as np


""" Functions to create hypothesis and references to apply NDCG """


def create_ndcg_inputs(predicted_probabilities,
                       ground_truth_probabilities,
                       ranked=True,
                       top_x=5,
                       top_y=5,
                       min_probability=0.05):
    """

    Args:
        predicted_probabilities: Numpy array of ML model probabilities.
        ground_truth_probabilities: Numpy array of ground truth probabilities.
        ranked: If True will compute ranked reference, else unranked.
        top_x: First X diagnosis in the sorted predicted probabilities used to build the hypothesis array.
        top_y: First Y diagnosis in the sorted ground truth probabilities used to build the references array.
        min_probability: Minimum probability of a ground truth diagnosis to be included inside the references, if it is
            smaller than that value would be considered as 0.


    Returns:
        references_scores: An array with all the ground truth reference scores
        references_indices: An array with the ground truth class indices
        hypothesis: An array with the predictions scores
    """
    if ranked:
        references_scores, references_indices = create_ranked_reference_from_probabilities(ground_truth_probabilities,
                                                                                           top_y,
                                                                                           min_probability
                                                                                           )
    else:
        references_scores, references_indices = create_unranked_reference_from_probabilities(ground_truth_probabilities,
                                                                                             top_y,
                                                                                             min_probability
                                                                                             )
    hypothesis = create_hypothesis(predicted_probabilities, references_indices, references_scores, top_x)

    return references_scores, references_indices, hypothesis


def create_unranked_reference_from_probabilities(probabilities, top_y, min_probability=0.05):
    """
    Create reference for unranked differentials
    """
    references_indices = []
    references_scores = []
    # Get CNN index sorted by max probability
    sorted_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, 0:top_y]
    for i, sample_indices in enumerate(sorted_indices):
        # Filter by min_probability
        reference_indices = np.array([index for index in sample_indices if probabilities[i, index] > min_probability])
        # Assign 1 to every reference
        reference_scores = np.ones((len(reference_indices)))
        # Padding scores with 0s and indices with -1s
        for k in range(len(reference_indices), top_y):
            reference_scores = np.append(reference_scores, 0)
            reference_indices = np.append(reference_indices, -1)
        references_indices.append(reference_indices)
        references_scores.append(reference_scores)
    return np.array(references_scores), np.array(references_indices)


def create_ranked_reference_from_probabilities(probabilities, top_y, min_probability=0.05):
    """
    Create reference for ranked differentials
    """
    references_indices = []
    references_scores = []
    # Get CNN index sorted by max probability
    sorted_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, 0:top_y]
    for i, sample_indices in enumerate(sorted_indices):
        # Filter by min_probability
        reference_indices = np.array([index for index in sample_indices if probabilities[i, index] > min_probability])
        reference_scores = np.array([probabilities[i, index] for index in reference_indices])
        # Normalize to sum to 1. Not necessarily needed but looks prettier visually
        reference_scores = reference_scores * 1 / sum(reference_scores)
        # Padding scores with 0s and indices with -1s
        for k in range(len(reference_indices), top_y):
            reference_scores = np.append(reference_scores, 0)
            reference_indices = np.append(reference_indices, -1)
        references_indices.append(reference_indices)
        references_scores.append(reference_scores)
    return np.array(references_scores), np.array(references_indices)


def create_hypothesis(probabilities, references_indices, references_scores, top_x):
    """
    Create hypothesis
    """
    top_x_hypothesis = []
    sorted_probability_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, 0:top_x]
    for i in range(len(references_indices)):
        hypothesis = []
        for index in sorted_probability_indices[i]:
            # If model index inside the references indices we copy the index reference score in the hypothesis array
            found_index = np.where(references_indices[i] == index)[0]
            if len(found_index) > 0 and probabilities[i, index] > 0:
                found_index = found_index[0]
                hypothesis.append(references_scores[i, found_index])
            else:
                hypothesis.append(0)
        top_x_hypothesis.append(hypothesis)
    return np.array(top_x_hypothesis)
