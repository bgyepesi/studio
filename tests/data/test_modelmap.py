import os
import numpy as np
import pandas as pd
import pytest

from backports.tempfile import TemporaryDirectory
from studio.data.snapshots import Snapshot
from studio.data.modelmap import ModelMap


def test_model_map(snapshot, ontology_tree):
    node_frequency = snapshot.compute_node_frequency()
    ontology_tree.set_node_count(node_frequency)

    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=0))

    conditions_df, diagnosis_df = model_map.conditions_df, model_map.diagnosis_df

    assert sorted(conditions_df.condition_name) == ['acne conglobata', 'acne excoriee des jeunes filles',
                                                    'acne fulminans', 'acne mechanica', 'acne vulgaris',
                                                    'childhood flexural comedones', 'follicular occlusion tetrad']

    assert sorted(conditions_df.n_samples) == [0, 0, 0, 0, 1, 2, 2]

    assert sorted(diagnosis_df.diagnosis_name) == ['acne vulgaris', 'childhood flexural comedones']
    assert sorted(diagnosis_df.malignancy) == ['benign', 'malignant']
    assert sorted(diagnosis_df.n_samples) == [2, 3]

    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=3))

    conditions_df, diagnosis_df = model_map.conditions_df, model_map.diagnosis_df

    assert sorted(conditions_df.condition_name) == ['acne conglobata', 'acne excoriee des jeunes filles',
                                                    'acne fulminans', 'acne mechanica', 'acne vulgaris',
                                                    'follicular occlusion tetrad']

    assert sorted(conditions_df.n_samples) == [0, 0, 0, 0, 1, 2]

    assert sorted(diagnosis_df.diagnosis_name) == ['acne vulgaris']
    assert sorted(diagnosis_df.malignancy) == ['benign']
    assert sorted(diagnosis_df.n_samples) == [3]


def test_save(snapshot, ontology_tree):
    node_frequency = snapshot.compute_node_frequency()
    ontology_tree.set_node_count(node_frequency)

    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=0))

    with TemporaryDirectory() as output_model_map_dir:
        filename = 'test_model_map_output'
        filepath = os.path.join(output_model_map_dir, filename)

        # CSV format
        format = 'csv'
        filepath_csv = filepath + '.' + format

        model_map.save(filepath, mode='conditions', format=format, columns=None)
        df = pd.read_csv(filepath_csv)
        assert len(df) == 7
        assert df.n_samples.tolist() == [1, 0, 0, 0, 2, 0, 2]
        model_map.save(filepath, mode='diagnosis', format=format, columns=None)
        df = pd.read_csv(filepath_csv)
        assert len(df) == 2
        assert df.n_samples.tolist() == [3, 2]
        assert df.diagnosis_id.tolist() == ['AIP:0002471', 'AIP:0002491']

        model_map.save(filepath, mode='diagnosis', format=format, columns=['diagnosis_id'])
        df = pd.read_csv(filepath_csv)
        assert len(df) == 2
        assert len(df.columns) == 1
        assert df.diagnosis_id.tolist() == ['AIP:0002471', 'AIP:0002491']

        # JSON format
        format = 'json'
        filepath_json = filepath + '.' + format
        model_map.save(filepath, mode='conditions', format=format, columns=None)
        df = pd.read_json(filepath_json)
        assert len(df) == 7
        assert df.n_samples.tolist() == [1, 0, 0, 0, 2, 0, 2]
        model_map.save(filepath, mode='diagnosis', format=format, columns=None)
        df = pd.read_json(filepath_json)
        assert len(df) == 2
        assert df.n_samples.tolist() == [3, 2]
        assert df.diagnosis_id.tolist() == ['AIP:0002471', 'AIP:0002491']
        model_map.save(filepath, mode='diagnosis', format=format, columns=['diagnosis_id'])
        df = pd.read_json(filepath_json)
        assert len(df) == 2
        assert len(df.columns) == 1
        assert df.diagnosis_id.tolist() == ['AIP:0002471', 'AIP:0002491']


def test_map_reviews(ontology_tree, snapshot, snapshot_reviews_manifest):
    # Manifest without reviews
    snapshot_manifest_df = snapshot.compute_average_reviews()
    ontology_tree.set_node_count(snapshot.compute_node_frequency())
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=0))
    mapped_reviews_df = model_map.map_reviews(snapshot_manifest_df,
                                              mode='reject_outliers')

    # Discard reviews without AIP code
    assert mapped_reviews_df.mapped_reviews.tolist() == [[{'diagnoses': {'AIP:0002491': 100}}],
                                                         [{'diagnoses': {'AIP:0002491': 100}}],
                                                         [{'diagnoses': {'AIP:0002471': 100}}],
                                                         [{'diagnoses': {'AIP:0002471': 100}}],
                                                         [{'diagnoses': {'AIP:0002471': 100}}]]

    # Manifest with reviews, keeping uncertainty and outliers as uncertainty
    snapshot = Snapshot(snapshot_manifest=snapshot_reviews_manifest)

    ontology_tree.set_node_count(snapshot.compute_node_frequency())
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=0))

    # Addition of nodes outside diagnosis node branches
    snapshot.manifest_df.at[4, 'reviews'] = \
        [
            {
                "reviewer_email": "1",
                "diagnoses": {'AIP:outside': 100}  # Node outside of a diagnosis node branch.
            },
            {
                "reviewer_email": "2",
                "diagnoses": {'AIP:0000001': 90, 'AIP:0002471': 10},  # Node above the diagnosis node.
            },
            {
                "reviewer_email": "3",
                "diagnoses": {'AIP:0002491': 30, 'AIP:0002471': 70}  # Nodes within the Model Map.
            }
    ]

    # Will convert reviews for outside diagnosis nodes to uncertainty
    mapped_reviews_df = model_map.map_reviews(snapshot.manifest_df,
                                              mode='outliers_as_uncertainty')

    assert mapped_reviews_df.mapped_reviews[4] == [{'reviewer_email': '1',
                                                    'diagnoses': {'uncertainty': 100}},
                                                   {'reviewer_email': '2',
                                                    'diagnoses': {'uncertainty': 90, 'AIP:0002471': 10}},
                                                   {'reviewer_email': '3',
                                                    'diagnoses': {'AIP:0002491': 30, 'AIP:0002471': 70}}]

    # Will reject the reviews that contain outside diagnosis nodes
    mapped_reviews_df = model_map.map_reviews(snapshot.manifest_df,
                                              mode='reject_outliers')

    assert mapped_reviews_df.mapped_reviews[4] == [{'reviewer_email': '3',
                                                    'diagnoses': {'AIP:0002491': 30, 'AIP:0002471': 70}}]


def test_map_snapshot_manifest_no_reviews(ontology_tree, snapshot):
    snapshot_manifest_df = snapshot.compute_average_reviews()
    ontology_tree.set_node_count(snapshot.compute_node_frequency())
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=1))

    # Manifest without reviews
    processed_manifest_df = model_map.map_manifest(snapshot_manifest_df,
                                                   mapping_mode='reject_outliers',)

    expected = [np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0]),
                np.array([1.0, 0.0])]
    np.testing.assert_array_almost_equal(processed_manifest_df['class_probabilities'].tolist(), expected)


def test_map_snapshot_manifest_reviews_uncertainty_as_class(ontology_tree, snapshot_reviews_manifest):
    # Manifest with reviews, keep uncertainty and add it as a class
    uncertainty_mode = 'keep'
    snapshot = Snapshot(snapshot_manifest=snapshot_reviews_manifest)
    ontology_tree.set_node_count(snapshot.compute_node_frequency(uncertainty_mode=uncertainty_mode))
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=1))

    processed_manifest_df = model_map.map_manifest(snapshot.manifest_df,
                                                   mapping_mode='outliers_as_uncertainty',
                                                   uncertainty_mode=uncertainty_mode,
                                                   uncertainty_as_class=True)

    # Uncertainty appears in model map
    assert model_map.diagnosis_df['diagnosis_id'].tolist() == ['AIP:0002471', 'AIP:0002491', 'uncertainty']

    expected = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.66, 0.18, 0.16], [0.2, 0.23, 0.57], [0.43, 0.23, 0.33]]
    np.testing.assert_array_almost_equal(processed_manifest_df['class_probabilities'].tolist(), expected, 2)

    # Manifest with reviews. Other uncertainty mode rather than keep will throw an error.
    uncertainty_mode = 'distribute'
    snapshot = Snapshot(snapshot_manifest=snapshot_reviews_manifest)
    ontology_tree.set_node_count(snapshot.compute_node_frequency(uncertainty_mode=uncertainty_mode))
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=1))

    with pytest.raises(ValueError, match='If `uncertainty_as_class` is True, `uncertainty_mode` must be `keep`'):
        model_map.map_manifest(snapshot.manifest_df,
                               mapping_mode='outliers_as_uncertainty',
                               uncertainty_mode=uncertainty_mode,
                               uncertainty_as_class=True)


def test_map_snapshot_manifest_reviews(ontology_tree, snapshot_reviews_manifest):
    # Manifest with reviews, Distribute uncertainty
    uncertainty_mode = 'distribute'
    snapshot = Snapshot(snapshot_manifest=snapshot_reviews_manifest)
    ontology_tree.set_node_count(snapshot.compute_node_frequency(uncertainty_mode=uncertainty_mode))
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=1))

    processed_manifest_df = model_map.map_manifest(snapshot.manifest_df,
                                                   mapping_mode='outliers_as_uncertainty',
                                                   uncertainty_mode=uncertainty_mode)

    expected = [[0.0, 1.0], [0.0, 1.0], [0.82, 0.18], [0.3, 0.7], [0.65, 0.35]]
    np.testing.assert_array_almost_equal(processed_manifest_df['class_probabilities'].tolist(), expected)

    # Manifest with reviews, drop uncertainty
    uncertainty_mode = 'drop'
    snapshot = Snapshot(snapshot_manifest=snapshot_reviews_manifest)
    ontology_tree.set_node_count(snapshot.compute_node_frequency(uncertainty_mode=uncertainty_mode))
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=1))

    processed_manifest_df = model_map.map_manifest(snapshot.manifest_df,
                                                   uncertainty_mode=uncertainty_mode,
                                                   mapping_mode='outliers_as_uncertainty')
    expected = [[0.0, 1.0], [0.0, 1.0], [0.7857, 0.2142], [0.4615, 0.5384], [0.65, 0.35]]
    np.testing.assert_array_almost_equal(processed_manifest_df['class_probabilities'].tolist(), expected, 4)

    # Manifest with reviews, keep uncertainty
    uncertainty_mode = 'keep'
    snapshot = Snapshot(snapshot_manifest=snapshot_reviews_manifest)

    ontology_tree.set_node_count(snapshot.compute_node_frequency(uncertainty_mode=uncertainty_mode))
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=1))

    processed_manifest_df = model_map.map_manifest(snapshot.manifest_df,
                                                   mapping_mode='reject_outliers',
                                                   uncertainty_threshold=0.5,
                                                   uncertainty_mode=uncertainty_mode)
    expected = [[0.0, 1.0], [0.0, 1.0], [0.74, 0.26], [0.6, 0.4]]
    np.testing.assert_array_almost_equal(processed_manifest_df['class_probabilities'].tolist(), expected, 4)

    # Manifest with reviews, will discard the reviews due to uncertainty being above the threshold
    uncertainty_mode = 'keep'
    snapshot = Snapshot(snapshot_manifest=snapshot_reviews_manifest)

    ontology_tree.set_node_count(snapshot.compute_node_frequency(uncertainty_mode=uncertainty_mode))

    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=1))

    processed_manifest_df = model_map.map_manifest(snapshot.manifest_df,
                                                   uncertainty_threshold=0.01,
                                                   uncertainty_mode=uncertainty_mode,
                                                   mapping_mode='outliers_as_uncertainty')

    expected = [[0.0, 1.0], [0.0, 1.0]]
    np.testing.assert_array_almost_equal(processed_manifest_df['class_probabilities'].tolist(), expected)


def test_map_snapshot_manifest_reviews_nodes_outside(ontology_tree, snapshot_reviews_manifest):
    # Manifest with reviews, will discard the reviews with codes outside the model map
    uncertainty_mode = 'keep'
    snapshot = Snapshot(snapshot_manifest=snapshot_reviews_manifest)

    snapshot.manifest_df.at[4, 'reviews'] = \
        [
            {
                "reviewer_email": "1",
                "diagnoses": {'AIP:outside': 100}  # Node outside of a diagnosis node branch.
            },
            {
                "reviewer_email": "2",
                "diagnoses": {'AIP:0000001': 90, 'AIP:0002471': 10},  # Node above the diagnosis node.
            },
            {
                "reviewer_email": "3",
                "diagnoses": {'AIP:0002491': 30, 'AIP:0002471': 70}  # Node within the Model Map.
            }
    ]

    ontology_tree.set_node_count(snapshot.compute_node_frequency(uncertainty_mode=uncertainty_mode))
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=0))

    processed_manifest_df = model_map.map_manifest(snapshot.manifest_df,
                                                   uncertainty_threshold=0.5,
                                                   uncertainty_mode=uncertainty_mode,
                                                   mapping_mode='outliers_as_uncertainty')

    # With mapping_mode='outliers_as_uncertainty'
    # Nodes outside and parents will be categorized as uncertainty. Uncertainty will be distributed among the nodes
    # outside the diagnosis nodes already in the average review. Last review will be removed due to high uncertainty
    # {'AIP:outside': 100} --> {'uncertainty': 100}
    # {'AIP:0000001': 90} --> {'uncertainty': 90}

    expected = [[0.0, 1.0], [0.0, 1.0], [0.74, 0.26]]
    np.testing.assert_array_almost_equal(processed_manifest_df['class_probabilities'].tolist(), expected, 4)

    # We skip the removal due to high uncertainty
    processed_manifest_df = model_map.map_manifest(snapshot.manifest_df,
                                                   uncertainty_threshold=1.1,
                                                   uncertainty_mode=uncertainty_mode,
                                                   mapping_mode='outliers_as_uncertainty')

    # Nodes outside and parents will be categorized as uncertainty. Uncertainty will be distributed among the nodes
    # outside the diagnosis nodes already in the average review.
    # {'AIP:outside': 100} --> {'AIP:0002491': 50, 'AIP:0002471': 50}
    # {'AIP:0000001': 90} --> {'AIP:0002491':90}

    expected = [[0.0, 1.0], [0.0, 1.0], [0.74, 0.26], [0.4833, 0.5166], [0.5833, 0.4166]]
    np.testing.assert_array_almost_equal(processed_manifest_df['class_probabilities'].tolist(), expected, 4)

    # With mapping_mode='reject_outliers'
    processed_manifest_df = model_map.map_manifest(snapshot.manifest_df,
                                                   uncertainty_threshold=1.1,
                                                   uncertainty_mode=uncertainty_mode,
                                                   mapping_mode='reject_outliers')

    # Reviews with nodes outside and DN ancestors will be rejected
    # {'AIP:outside': 100} --> {}
    # {'AIP:0000001': 90} --> {}
    # Probabilities of review 5 will be the ones from the last review
    expected = [[0.0, 1.0], [0.0, 1.0], [0.74, 0.26], [0.4833, 0.5166], [0.7, 0.3]]
    np.testing.assert_array_almost_equal(processed_manifest_df['class_probabilities'].tolist(), expected, 4)

    # With mapping_mode='reject_outliers'
    processed_manifest_df = model_map.map_manifest(snapshot.manifest_df,
                                                   uncertainty_threshold=1.1,
                                                   uncertainty_mode=uncertainty_mode,
                                                   min_reviews=2,
                                                   mapping_mode='reject_outliers')
    # Reviews with nodes outside and DN ancestors will be rejected
    # {'AIP:outside': 100} --> {}
    # {'AIP:0000001': 90} --> {}
    # Review 5 will be rejected as it only has 1 review, while min_reviews parameter is 2
    expected = [[0.0, 1.0], [0.0, 1.0], [0.74, 0.26], [0.4833, 0.5166]]
    np.testing.assert_array_almost_equal(processed_manifest_df['class_probabilities'].tolist(), expected, 4)


def test_model_map_diagnosis_df():
    # Three conditions mapping to two diagnosis nodes.
    condition_dict = {
        'class_index': [1, 1, 0],  # The nodes indexes of the CNN.
        'condition_id': ['100', '303', '123'],  # The IDs of the conditions.
        'condition_name': ['acne variant', 'acne', 'melanoma'],  # Condition names.
        'diagnosis_id': ['222', '222', '111'],  # First two conditions map to the diagnosis id 222.
        'diagnosis_name': ['acne', 'acne', 'melanoma'],  # Diagnosis node names.
        'malignancy': ['benign', 'benign', 'malignant'],  # Malignancy of the condition.
        'n_samples': [5, 7, 10]  # Number of samples (images) for the condition.
    }
    condition_df = pd.DataFrame(condition_dict)

    mm = ModelMap(conditions_df=condition_df)
    diagnosis_dict = mm.diagnosis_df.to_dict('list')

    # Two diagnosis nodes.
    expected_diagnosis_dict = {
        'class_index': [0, 1],  # Node indexes of the CNN.
        'diagnosis_id': ['111', '222'],
        'diagnosis_name': ['melanoma', 'acne'],
        'malignancy': ['malignant', 'benign'],
        'condition_id': [['123'], ['100', '303']],  # A list of conditions IDs that map to the diagnosis node.
        'condition_name': [['melanoma'], ['acne variant', 'acne']],
        'n_samples': [10, 12]  # Total number of samples that map to the diagnosis node.
    }

    assert diagnosis_dict == expected_diagnosis_dict, diagnosis_dict

    # Change a condition to be malignant.
    condition_dict['malignancy'][0] = 'malignant'
    condition_df = pd.DataFrame(condition_dict)

    with pytest.raises(Exception):
        # This fails since a diagnosis node maps to a condition that is benign and another that is malignant.
        # Thus an additional row will be added to the diagnosis_df that does not correspond to a unique class index.
        ModelMap(conditions_df=condition_df)
