from studio.data.review_algorithms import uncertainty_is_above_threshold, compute_average_review, reject_outliers, \
    outliers_as_uncertainty, keep_outliers, distribute_ancestors_to_dn_outliers_to_uncertainty
from studio.data.modelmap import ModelMap


def test_compute_average_review(review_list):
    uncertainty_mode = 'keep'
    average_review = compute_average_review(review_list[0], uncertainty_mode)

    assert average_review == {
        'AIP:0002471': 0.54,
        'AIP:0002480': 0.12,
        'uncertainty': 0.16,
        'AIP:0002481': 0.18
    }
    assert sum(average_review.values()) == 1.0

    average_review = compute_average_review(review_list[1], uncertainty_mode)
    assert average_review == {
        'uncertainty': 0.5666666666666667,
        'AIP:0002481': 0.23333333333333334,
        'AIP:0002471': 0.2
    }
    assert sum(average_review.values()) == 1.0

    uncertainty_mode = 'distribute'

    average_review = compute_average_review(review_list[0], uncertainty_mode)
    assert average_review == {
        'AIP:0002471': 0.68,
        'AIP:0002480': 0.14,
        'AIP:0002481': 0.18
    }
    assert sum(average_review.values()) == 1.0

    # Will drop the 100% uncertainty and distribute when there is more than 1 diagnosis per review
    """
    [{'uncertainty': 100},
    {'AIP:0002481': 40, 'AIP:0002471': 60},
    {'AIP:0002481': 30, 'uncertainty': 70}],
    """

    average_review = compute_average_review(review_list[1], uncertainty_mode)
    assert average_review == {
        'AIP:0002481': 0.7,
        'AIP:0002471': 0.3
    }
    assert sum(average_review.values()) == 1.0

    uncertainty_mode = 'drop'

    # Will drop the uncertainty value, sum confidence values and then normalize

    average_review = compute_average_review(review_list[1], uncertainty_mode)
    assert average_review == {
        'AIP:0002481': 0.5384615384615384,
        'AIP:0002471': 0.46153846153846156
    }
    assert sum(average_review.values()) == 1.0


def test_uncertainty_is_above_threshold(review_list):
    expected = [False, True, False]
    for i, review in enumerate(review_list):
        assert uncertainty_is_above_threshold(review, threshold=0.5) == expected[i]

    assert uncertainty_is_above_threshold(review_list[2], threshold=0.1)


def test_reject_outliers(ontology_tree):
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=0))
    # Diagnosis node in review, nothing changes
    diagnoses = {'AIP:0002471': 50, 'uncertainty': 50}
    mapped_diagnoses = reject_outliers(diagnoses, model_map)
    assert mapped_diagnoses == {'AIP:0002471': 50, 'uncertainty': 50}
    # Diagnosis node child, will be aggregated to parent
    diagnoses = {'AIP:0002480': 100}
    mapped_diagnoses = reject_outliers(diagnoses, model_map)
    assert mapped_diagnoses == {'AIP:0002471': 100}
    # Node outside, will have empty diagnoses
    diagnoses = {'AIP:0002471': 10, 'outside': 90}
    mapped_diagnoses = reject_outliers(diagnoses, model_map)
    assert mapped_diagnoses == {}


def test_outliers_as_uncertainty(ontology_tree):
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=0))
    # Diagnosis node in review, nothing changes
    diagnoses = {'AIP:0002471': 50, 'uncertainty': 50}
    mapped_diagnoses = outliers_as_uncertainty(diagnoses, model_map)
    assert mapped_diagnoses == {'AIP:0002471': 50, 'uncertainty': 50}
    # Diagnosis node child, will be aggregated to parent
    diagnoses = {'AIP:0002480': 100}
    mapped_diagnoses = outliers_as_uncertainty(diagnoses, model_map)
    assert mapped_diagnoses == {'AIP:0002471': 100}
    # Node outside or ancestors, will add uncertainty
    diagnoses = {'AIP:0002480': 10, 'outside': 60, 'AIP:0000007': 10, 'uncertainty': 20}
    mapped_diagnoses = outliers_as_uncertainty(diagnoses, model_map)
    assert mapped_diagnoses == {'AIP:0002471': 10, 'uncertainty': 90}


def test_keep_outliers(ontology_tree):
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=0))
    # Diagnosis node in review, nothing changes
    diagnoses = {'AIP:0002471': 50, 'uncertainty': 50}
    mapped_diagnoses = keep_outliers(diagnoses, model_map)
    assert mapped_diagnoses == {'AIP:0002471': 50, 'uncertainty': 50}
    # Diagnosis node child, will be aggregated to parent
    diagnoses = {'AIP:0002480': 100}
    mapped_diagnoses = keep_outliers(diagnoses, model_map)
    assert mapped_diagnoses == {'AIP:0002471': 100}
    # Node outside or ancestors, will leave uncertainty
    diagnoses = {'AIP:0002480': 10, 'outside': 60, 'AIP:0000007': 10, 'uncertainty': 20}
    mapped_diagnoses = keep_outliers(diagnoses, model_map)
    assert mapped_diagnoses == {'AIP:0002471': 10, 'outside': 60, 'AIP:0000007': 10, 'uncertainty': 20}


def test_distribute_ancestors_to_dn_outliers_to_uncertainty(ontology_tree):
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=0))
    ancestors_diagnosis_ids_map = ontology_tree.get_ancestors_diagnosis_ids_map()
    mapping_function = distribute_ancestors_to_dn_outliers_to_uncertainty(ancestors_diagnosis_ids_map)
    # Diagnosis node in review, nothing changes
    diagnoses = {'AIP:0002471': 50, 'uncertainty': 50}
    mapped_diagnoses = mapping_function(diagnoses, model_map)
    assert mapped_diagnoses == {'AIP:0002471': 50, 'uncertainty': 50}
    # Diagnosis node child, will be aggregated to parent
    diagnoses = {'AIP:0002480': 100}
    mapped_diagnoses = mapping_function(diagnoses, model_map)
    assert mapped_diagnoses == {'AIP:0002471': 100}
    # Node outside will be converted to uncertainty, Ancestor ('AIP:0000007') will be distributed between its two
    # DN descendants 'AIP:0002471' and 'AIP:0002491'
    diagnoses = {'AIP:0002480': 10, 'outside': 60, 'AIP:0000007': 10, 'uncertainty': 20}
    mapped_diagnoses = mapping_function(diagnoses, model_map)
    assert mapped_diagnoses == {'AIP:0002471': 15, 'AIP:0002491': 5, 'uncertainty': 80}
