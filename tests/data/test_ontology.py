import re
import json
import pytest
import pandas as pd

from copy import deepcopy
from studio.data.ontology import Concept, Ontology


def _create_example_dataframe():
    """Return a dataframe with example values.
    The dataframe has 3 nodes: a root node and two child nodes, where both children point to the root.
    =====
    df = _create_example_dataframe()
    print(df)
              id         label         type    malignancy   parent_id
    0    root_id    root_label    root_type    root_malig
    1  child1_id  child1_label  child1_type  child1_malig   root_id
    2  child2_id  child2_label  child2_type  child2_malig   root_id
    =====
    """

    node_ids = ['root_id', 'child1_id', 'child2_id']
    node_labels = ['root_label', 'child1_label', 'child2_label']
    node_types = ['root_type', 'child1_type', 'child2_type']
    node_malig = ['unspecified', 'benign', 'malignant']
    node_show_during_review = [True, True, False]
    parent_ids = ['', 'root_id', 'root_id']
    df = pd.DataFrame({
        Ontology.NODE_ID: node_ids,
        Ontology.NODE_LABEL: node_labels,
        Ontology.NODE_TYPE: node_types,
        Ontology.NODE_MALIGNANCY: node_malig,
        Ontology.NODE_PARENT_ID: parent_ids,
        Ontology.NODE_SHOW_DURING_REVIEW: node_show_during_review,
    })
    return df


def test_concept():
    """Test ontology.Concept's class"""
    concept = Concept(images=1, aggregated_images=2, node_type='general', malignancy='benign')
    assert concept.images == 1
    assert concept.aggregated_images == 2
    assert concept[Ontology.NODE_TYPE] == 'general'
    assert concept.malignancy == 'benign'


def test_build(ontology_tree):
    """Test ontology.Ontology's build() function."""
    assert ontology_tree.size() == len(ontology_tree.all_nodes()), ontology_tree.size()
    assert ontology_tree.get_node('AIP:0000000').data[Ontology.NODE_TYPE] == 'general'
    assert ontology_tree.get_node('AIP:0002471').data[Ontology.NODE_TYPE] == 'diagnosis'
    assert ontology_tree.get_node('AIP:0000001').data.malignancy == 'unspecified'
    assert ontology_tree.get_node('AIP:0100001').data.malignancy == 'malignant'


def test_ontology_get_node_summary():
    """Test Ontology.get_node_summary()"""

    nodes = pd.DataFrame({
        Ontology.NODE_ID: ['node_id'],
        Ontology.NODE_LABEL: ['node_label'],
        Ontology.NODE_TYPE: ['diagnosis'],
        Ontology.NODE_MALIGNANCY: ['malignant'],
    })

    node = nodes.iloc[0]  # Get the first and only entry.
    node_summary = Ontology.get_node_summary(node)
    # The node_summary should have this form.
    assert node_summary == 'node_label [node_id](M)*', node_summary

    node.type = 'general'
    node.malignancy = 'benign'
    node_summary = Ontology.get_node_summary(node)
    # The node_summary no longer has the * to represent the diagnosis node and shows a (B) for benign.
    assert node_summary == 'node_label [node_id](B)', node_summary


def test_get_name(ontology_tree):
    """Test ontology.Ontology's get_name() function."""
    assert ontology_tree.get_name('AIP:0002471') == 'acne vulgaris'


def test_get_id(ontology_tree):
    """Test ontology.Ontology's get_id() function"""
    assert ontology_tree.get_id('acne vulgaris') == 'AIP:0002471'


def test_get_images(ontology_tree, node_frequency_counts):
    """Test ontology.Ontology's get_images() function"""
    ontology_tree.set_node_count(node_frequency_counts)
    assert ontology_tree.get_images('AIP:root') == 0
    assert ontology_tree.get_images('AIP:0002471') == 1
    assert ontology_tree.get_images('AIP:0100001') == 0


def test_get_aggregated_images(ontology_tree, node_frequency_counts):
    """Test ontology.Ontology's get_images() function"""
    ontology_tree.set_node_count(node_frequency_counts)
    assert ontology_tree.get_aggregated_images('AIP:root') == 8
    assert ontology_tree.get_aggregated_images('AIP:0002471') == 3
    assert ontology_tree.get_aggregated_images('AIP:0100001') == 5


def test_get_malignancy(ontology_tree):
    """Test ontology.Ontology get_id() function"""
    assert ontology_tree.get_malignancy('AIP:0002471') == 'benign'
    assert ontology_tree.get_malignancy('AIP:0100001') == 'malignant'


def test_sort_by(ontology_tree):
    """Test ontology.Ontology's sort_by() function"""
    nodes = ['AIP:0002471', 'AIP:0000001', 'AIP:0000000']
    nodes_by_level = ontology_tree.sort_by(nodes)
    assert nodes_by_level == ['AIP:0000000', 'AIP:0000001', 'AIP:0002471'], nodes_by_level


def test_get_internal_nodes(ontology_tree):
    """Test ontology.Ontology's get_internal_nodes() function"""
    internal_nodes = ontology_tree.get_internal_nodes()
    for node in internal_nodes:
        assert len(ontology_tree.children(node.identifier)) > 0, "all internal nodes contain children"


def test_remove_nodes(ontology_tree):
    """Test ontology.Ontology's remove_nodes() function"""
    tree = deepcopy(ontology_tree)
    nodes_to_remove = ['AIP:0100001', 'AIP:0002471']
    for node in nodes_to_remove:
        assert tree.contains(node)

    tree.remove_nodes(nodes_to_remove)
    for node in nodes_to_remove:
        assert not tree.contains(node)


def test_connected_node_ids(ontology_tree):
    """Test ontology.Ontology's connected_node_ids() function"""
    node_ids = ontology_tree.connected_node_ids(['AIP:0000001'])
    assert sorted(node_ids) == sorted(['AIP:0000001', 'AIP:0000000', 'AIP:root']), sorted(node_ids)


def test_remove_unconnected_nodes(ontology_tree):
    tree = deepcopy(ontology_tree)
    original_len = len(tree)
    keep_node_ids = ['AIP:0002481']
    n_removed = tree.remove_unconnected_nodes(keep_node_ids)
    actual_nodes = sorted([node_id for node_id in tree.nodes])
    expected_nodes = sorted(['AIP:root', 'AIP:0000000', 'AIP:0000001', 'AIP:0000007', 'AIP:0002471',
                             'AIP:0002480', 'AIP:0002481'])
    assert actual_nodes == expected_nodes, actual_nodes
    assert n_removed == original_len - len(actual_nodes), n_removed


def test_not_leaf_nodes(ontology_tree):
    """Test ontology.Ontology's not_leaf_nodes() function"""
    nodes = ontology_tree.not_leaf_nodes()
    not_leaf_ids = [node.identifier for node in nodes]

    for node in ontology_tree.all_nodes():
        if node.is_leaf():
            assert node.identifier not in not_leaf_ids
        else:
            assert node.identifier in not_leaf_ids


def test_nodes_to_root(ontology_tree):
    """Test ontology.ontolgy.Ontology's nodes_to_root() function"""
    nodes = ontology_tree.nodes_to_root('AIP:0002471')
    node_ids = [node.identifier for node in nodes]
    expected_node_ids = ['AIP:0002471', 'AIP:0000007', 'AIP:0000001', 'AIP:0000000', 'AIP:root']
    assert node_ids == expected_node_ids


def test_get_descendants(ontology_tree):
    """Test ontology.Ontology's get_descendants() function"""
    actual = ontology_tree.get_descendants('AIP:0002471')
    expected = ['AIP:0002471', 'AIP:0002480', 'AIP:0002481', 'AIP:0002484', 'AIP:0002478',
                'AIP:0001379']
    assert actual == expected


def test_get_diagnosis_ids(ontology_tree):
    """Test ontology.Ontology's get_diagnosis_nodes() function"""
    diagnosis_ids = ontology_tree.get_diagnosis_ids()
    assert diagnosis_ids[0] == 'AIP:0002471', diagnosis_ids[0]


def test_branch_node_ids(ontology_tree):
    """Test ontology.Ontology's branch_node_ids() function"""
    actual = sorted(ontology_tree.branch_node_ids('AIP:0002480'))
    expected = sorted(['AIP:0002480', 'AIP:0002481', 'AIP:0000001', 'AIP:0002471', 'AIP:0000000',
                       'AIP:0000007', 'AIP:root'])
    assert actual == expected


def test_node_ids_of_type(ontology_tree):
    """Test ontology.Ontology's node_ids_of_type() function"""
    node_ids = ['AIP:0000001', 'AIP:0000007', 'AIP:0002471', 'AIP:0002480']
    actual = ontology_tree.node_ids_of_type(node_ids, 'diagnosis')
    expected = ['AIP:0002471']
    assert actual == expected, 'only AIP:0002471 from node_ids is diagnosis'

    actual = sorted(ontology_tree.node_ids_of_type(node_ids, 'general'))
    expected = sorted(['AIP:0000001', 'AIP:0000007', 'AIP:0002480'])
    assert actual == expected, 'the rest are general type nodes'


def test_nodes_coverage_by_diagnosis(ontology_tree):
    """Test ontology.Ontology's nodes_coverage_by_diagnosis() function"""
    covered_ids, not_covered_ids = ontology_tree.nodes_coverage_by_diagnosis()
    expected_covered_ids = sorted(['AIP:root', 'AIP:0000000', 'AIP:0000001', 'AIP:0000007',
                                   'AIP:0002471', 'AIP:0002480', 'AIP:0002484', 'AIP:0002478',
                                   'AIP:0001379', 'AIP:0002475', 'AIP:0002481', 'AIP:0002491',
                                   'AIP:0100001'])
    assert sorted(covered_ids) == sorted(expected_covered_ids), 'descendant nodes from AIP:0002471 are covered'

    expected_not_covered_ids = list(set(ontology_tree.nodes) - set(expected_covered_ids))
    assert sorted(not_covered_ids) == sorted(expected_not_covered_ids), 'difference between all nodes and covered nodes'


def test_reset_node_count(ontology_tree):
    """Test ontology.Ontology's _reset_node_count() function"""
    tree = deepcopy(ontology_tree)
    tree._reset_node_count(mode='images')
    for node in tree.all_nodes():
        assert node.data.images == 0

    tree._reset_node_count(mode='aggregated_images')
    for node in tree.all_nodes():
        assert node.data.aggregated_images == 0


def test_set_node_count(ontology_tree):
    """Test ontology.Ontology's initialize_image_count() function"""
    nodes_frequency = pd.DataFrame({'node_id': ['AIP:root',
                                                'AIP:0002471',
                                                'AIP:0002481',
                                                'AIP:0002478',
                                                'AIP:0002491'],
                                    'frequency': [0, 1, 2, 2, 5]})
    ontology_tree.set_node_count(nodes_frequency)

    assert ontology_tree.get_images('AIP:root') == 0
    assert ontology_tree.get_images('AIP:0002471') == 1
    assert ontology_tree.get_images('AIP:0002481') == 2
    assert ontology_tree.get_images('AIP:0002478') == 2
    assert ontology_tree.get_images('AIP:0002491') == 5
    assert ontology_tree.get_images('AIP:0100001') == 0

    assert ontology_tree.get_aggregated_images('AIP:root') == 10
    assert ontology_tree.get_aggregated_images('AIP:0002471') == 5
    assert ontology_tree.get_aggregated_images('AIP:0002481') == 2
    assert ontology_tree.get_aggregated_images('AIP:0002478') == 2
    assert ontology_tree.get_aggregated_images('AIP:0002491') == 5
    assert ontology_tree.get_aggregated_images('AIP:0100001') == 5


def test_aggregate_images(ontology_tree):
    """Test ontology.Ontology's aggregate_images() function"""
    ontology_tree._reset_node_count('images')
    ontology_tree._reset_node_count('aggregated_images')

    # Check original aggregation of images from AIP:0002481 to AIP:root
    assert ontology_tree.get_node('AIP:0002481').data.aggregated_images == 0
    assert ontology_tree.get_node('AIP:0002480').data.aggregated_images == 0
    assert ontology_tree.get_node('AIP:0002471').data.aggregated_images == 0
    assert ontology_tree.get_node('AIP:0000007').data.aggregated_images == 0
    assert ontology_tree.get_node('AIP:0000001').data.aggregated_images == 0
    assert ontology_tree.get_node('AIP:0000000').data.aggregated_images == 0
    assert ontology_tree.get_node('AIP:root').data.aggregated_images == 0

    # Increase AIP:0002481's images by one
    ontology_tree.get_node('AIP:0002481').data.images += 1
    ontology_tree.aggregate_images()

    # Assert new image aggregation count is the previous one +1
    assert ontology_tree.get_node('AIP:0002491').data.aggregated_images == 0
    assert ontology_tree.get_node('AIP:0002481').data.aggregated_images == 1
    assert ontology_tree.get_node('AIP:0002480').data.aggregated_images == 1
    assert ontology_tree.get_node('AIP:0002471').data.aggregated_images == 1
    assert ontology_tree.get_node('AIP:0000007').data.aggregated_images == 1
    assert ontology_tree.get_node('AIP:0000001').data.aggregated_images == 1
    assert ontology_tree.get_node('AIP:0000000').data.aggregated_images == 1
    assert ontology_tree.get_node('AIP:root').data.aggregated_images == 1


def test_nodes_by_level(ontology_tree):
    """Test ontology.Ontology's nodes_by_level() function"""
    level_nodes = ontology_tree.nodes_by_level()
    assert level_nodes[0] == ['AIP:root'], 'nodes from level 0'
    assert level_nodes[1] == ['AIP:0000000'], 'nodes from level 1'
    assert level_nodes[2] == ['AIP:0000001'], 'nodes from level 2'
    assert level_nodes[3] == ['AIP:0000007'], 'nodes from level 3'
    assert level_nodes[4] == ['AIP:0002471', 'AIP:0100001'], 'nodes from level 4'
    assert level_nodes[5] == ['AIP:0002480', 'AIP:0002484', 'AIP:0002478', 'AIP:0001379',
                              'AIP:0002475'], 'nodes from level 5'
    assert level_nodes[6] == ['AIP:0002481', 'AIP:0002491'], 'nodes from level 6'


def test_prune_nodes_by_image(ontology_tree, node_frequency_counts):
    """Test ontology.Ontology's prune_nodes_by_image() function"""

    node_ids = ontology_tree.get_descendants('AIP:0002471')

    ontology_tree.set_node_count(node_frequency_counts)

    tree = deepcopy(ontology_tree)

    assert tree.prune_nodes_by_image(node_ids, 1) == 4, 'four nodes with less than 1 images'

    tree = deepcopy(ontology_tree)
    assert tree.prune_nodes_by_image(node_ids, 2) == 5, 'five nodes with less than 2 images'

    tree = deepcopy(ontology_tree)
    assert tree.prune_nodes_by_image(node_ids, 4) == 6, 'six nodes with less than 4 images'

    tree = deepcopy(ontology_tree)
    assert tree.prune_nodes_by_image(node_ids, 5) == 6, 'six nodes with less than 5 images'


def test_filter_min_images(ontology_tree, node_frequency_counts):
    """Test ontology.Ontology's filter_min_images() function"""
    ontology_tree.set_node_count(node_frequency_counts)

    node_ids = ontology_tree.get_descendants('AIP:0002471')

    actual = sorted(ontology_tree.filter_min_images(node_ids, 1))
    expected = sorted(['AIP:0002471', 'AIP:0002478'])
    assert actual == expected, 'descendant nodes of AIP:0002471 with more than 1 images'

    actual = sorted(ontology_tree.filter_min_images(node_ids, 2))
    expected = sorted(['AIP:0002478'])
    assert actual == expected, 'descendant nodes of AIP:0002471 with more than 2 images'

    actual = sorted(ontology_tree.filter_min_images(node_ids, 3))
    expected = []
    assert actual == expected, 'descendant nodes of AIP:0002471 with more than 3 images'


def test_filter_min_aggregated_images(ontology_tree, node_frequency_counts):
    """Test ontology.Ontology's filter_min_aggregated_images() function"""
    node_ids = ontology_tree.get_descendants('AIP:0000007')
    ontology_tree.set_node_count(node_frequency_counts)

    actual = sorted(ontology_tree.filter_min_aggregated_images(node_ids, 1))
    expected = sorted(['AIP:0000007', 'AIP:0002471', 'AIP:0002475', 'AIP:0002478', 'AIP:0002491',
                       'AIP:0100001'])
    assert actual == expected, 'descendant nodes of AIP:0000007 with more than 1 aggregated images'

    actual = sorted(ontology_tree.filter_min_aggregated_images(node_ids, 3))
    expected = sorted(['AIP:0000007', 'AIP:0002471', 'AIP:0002475', 'AIP:0002491', 'AIP:0100001'])
    assert actual == expected, 'descendant nodes of AIP:0000007 with more than 2 aggregated images'

    actual = sorted(ontology_tree.filter_min_aggregated_images(node_ids, 5))
    expected = sorted(['AIP:0000007', 'AIP:0002475', 'AIP:0002491', 'AIP:0100001'])
    assert actual == expected, 'descendant nodes of AIP:0000007 with more than 5 images'


def test_to_dataframe(ontology_tree):
    """Test ontology.Ontology's to_dataframe() function"""
    # pd.util.testing.assert_frame_equal(ontology_tree.df, ontology_tree.to_dataframe())
    tree = deepcopy(ontology_tree)
    tree.remove_nodes(['AIP:0000007'])
    assert len(tree.to_dataframe()) == 3, tree.to_dataframe()


def test_to_json(ontology_tree):
    """Test ontology.Ontology's to_json() function"""
    tree_json = ontology_tree.to_json()
    dict_json = json.loads(tree_json)
    assert len(dict_json['nodes']) == len(ontology_tree), len(dict_json['nodes'])
    assert len(dict_json['edges']) == 12, len(dict_json['edges'])


def test_to_jstree_json(ontology_tree):
    """Test ontology.Ontology's to_jstree_json() function"""
    jstree_json = ontology_tree.to_jstree_json()
    dict_jstree_json = json.loads(jstree_json)
    # Check that all nodes have all the expected properties
    for node in dict_jstree_json:
        assert sorted([Ontology.NODE_ID, 'parent', 'text', 'state', 'data']) == sorted(node.keys())
        assert sorted([Ontology.NODE_LABEL, Ontology.NODE_TYPE, Ontology.NODE_IMAGES, Ontology.NODE_AGG_IMAGES,
                       Ontology.NODE_MALIGNANCY, Ontology.NODE_SHOW_DURING_REVIEW]) == sorted(node['data'].keys())


def test_json_nodes_from_dataframe(json_ontology):
    # Get the test dataframe.
    df = _create_example_dataframe()
    # Get expected values.
    expected_json_nodes = json_ontology[Ontology.NODES]

    # Static function.
    json_nodes = Ontology.json_nodes_from_dataframe(df)

    # Check all the nodes are in the expected format.
    assert json_nodes == expected_json_nodes, json_nodes


def test_json_edges_from_dataframe(json_ontology):
    # Get the test dataframe.
    df = _create_example_dataframe()
    expected_json_edges = json_ontology[Ontology.EDGES]

    # Static function.
    json_edges = Ontology.json_edges_from_dataframe(df)

    # Check the expected edges are returned.
    assert json_edges == expected_json_edges, json_edges


def test_json_ontology_from_dataframe(json_ontology):
    # Get the test dataframe.
    df = _create_example_dataframe()
    expected_json_ontology = json_ontology

    actual_json_ontology = Ontology.json_ontology_from_dataframe(df)
    assert actual_json_ontology == expected_json_ontology


def test_get_ordered_node_ids(json_ontology):
    """Ensure the node IDs are returned in the expected format."""
    tree = Ontology(json_ontology, root_id='root_id')

    tree_order = tree.get_ordered_node_ids(order_by='tree-level')
    # `root_id` is the root. The next two are sorted by alphanumeric IDs.
    assert tree_order == ['root_id', 'child1_id', 'child2_id'], tree_order

    id_order = tree.get_ordered_node_ids(order_by='identifier')
    # `child1_id` is the first alphanumeric ID.
    assert id_order == ['child1_id', 'child2_id', 'root_id'], id_order


def test_json_node_from_node_data():
    """Test only the relevant data is exported to a JSON node."""

    node_data = {
        Ontology.NODE_ID: 'node_id',
        Ontology.NODE_LABEL: 'node_label',
        Ontology.NODE_IMAGES: 123,  # Images now decoupled from ontology. Thus ignored in the JSON.
        Ontology.NODE_TYPE: 'node_type',
        Ontology.NODE_MALIGNANCY: 'node_mal',
        Ontology.NODE_SHOW_DURING_REVIEW: True,
        'IGNORE_KEY': 'ignore_key',  # Extra value we do not want in the outputted JSON.
    }

    expected_json_node = {
        Ontology.NODE_ID: 'node_id',
        Ontology.NODE_LABEL: 'node_label',
        Ontology.NODE_TYPE: 'node_type',
        Ontology.NODE_MALIGNANCY: 'node_mal',
        Ontology.NODE_SHOW_DURING_REVIEW: True
    }

    json_node = Ontology.json_node_from_node_data(node_data)
    assert json_node == expected_json_node, json_node


def test_check_diagnosis_nodes_collisions_with_descendants(ontology_manifest_errors_dn):
    ontology = Ontology(ontology_manifest_errors_dn)
    errors_report = ontology.errors_report
    assert errors_report['Error type'].tolist() == ['DN Uniqueness']
    assert errors_report['Nodes'].tolist() == ['AIP:0002471']
    assert errors_report['Collisions'].tolist() == [['AIP:0002480', 'AIP:0001379']]


def test_check_same_malignancy_in_descendants(ontology_manifest_errors_malignancy):
    ontology = Ontology(ontology_manifest_errors_malignancy)
    errors_report = ontology.errors_report
    assert errors_report['Error type'].tolist() == ['Different Malignancy', 'Different Malignancy']
    assert errors_report['Nodes'].tolist() == ['AIP:0002475', 'AIP:0100001']
    assert errors_report['Collisions'].tolist() == [['AIP:0002491'], ['AIP:0002475']]


def test_check_duplicate_ids(ontology_manifest_errors_duplicate_ids):
    with pytest.raises(Exception,
                       match=re.escape("Can't create node with ID 'AIP:0002471")):
        Ontology(ontology_manifest_errors_duplicate_ids)


def test_check_label_uniqueness(ontology_manifest_errors_duplicate_labels):
    ontology = Ontology(ontology_manifest_errors_duplicate_labels)
    errors_report = ontology.errors_report
    assert errors_report['Error type'].tolist() == ['Duplicate Labels']
    assert errors_report['Nodes'].tolist() == [[]]
    assert errors_report['Collisions'].tolist() == [['acne vulgaris', 'adnexal disease']]


def test_check_multiple_errors(ontology_manifest_multiple_errors):
    ontology = Ontology(ontology_manifest_multiple_errors)
    errors_report = ontology.errors_report
    assert len(errors_report) == 4
    assert errors_report['Error type'].tolist() == ['Duplicate Labels', 'DN Uniqueness', 'Different Malignancy',
                                                    'Different Malignancy']
    assert errors_report['Nodes'].tolist() == [[], 'AIP:0002471', 'AIP:0002475', 'AIP:0100001']
    assert errors_report['Collisions'].tolist() == [['acne vulgaris', 'adnexal disease'],
                                                    ['AIP:0002480', 'AIP:0001379'],
                                                    ['AIP:0002491'], ['AIP:0002475']]


def test_set_diagnosis_nodes(ontology_tree):
    missing_nodes = ontology_tree.set_diagnosis_nodes(['AIP:0002491',
                                                       'AIP:0002480',
                                                       'AIP:0002481',
                                                       'Error'])
    assert missing_nodes == ['Error']

    errors_report = ontology_tree.errors_report
    assert errors_report['Error type'].tolist() == ['DN Uniqueness']
    assert errors_report['Nodes'].tolist() == ['AIP:0002480']
    assert errors_report['Collisions'].tolist() == [['AIP:0002481']]


def test_compute_conditions_df(json_ontology):
    # Initialize ontology.
    tree = Ontology(json_ontology, root_id='root_id')

    # Set diagnosis nodes.
    tree.set_diagnosis_nodes(['child1_id', 'child2_id'])

    # Set number of images associated with each node.
    tree.set_node_count(pd.DataFrame({'node_id': ['child1_id', 'child2_id'], 'frequency': [1, 2]}))
    tree.aggregate_images()

    # Min images = 2, but force the `child1_id` to be included, even though it does not meet the threshold.
    conditions_df = tree.compute_conditions_df(min_diagnosis_images=2, force_diagnosis_ids=['child1_id'])
    actual_diagnosis_nodes = sorted(conditions_df['diagnosis_id'].tolist())
    actual_conditions_nodes = sorted(conditions_df['condition_id'].tolist())
    assert actual_diagnosis_nodes == ['child1_id', 'child2_id'], actual_diagnosis_nodes
    assert actual_conditions_nodes == ['child1_id', 'child2_id'], actual_conditions_nodes

    # Even `child2_id` has enough images to be inside, as it is not in the constrain_diagnosis_ids it will be exluded
    conditions_df = tree.compute_conditions_df(min_diagnosis_images=1, constrain_diagnosis_ids=['child1_id'])
    actual_diagnosis_nodes = sorted(conditions_df['diagnosis_id'].tolist())
    actual_conditions_nodes = sorted(conditions_df['condition_id'].tolist())
    assert actual_diagnosis_nodes == ['child1_id'], actual_diagnosis_nodes
    assert actual_conditions_nodes == ['child1_id'], actual_conditions_nodes

    # Initialize ontology.
    tree = Ontology(json_ontology, root_id='root_id')

    # Set a single diagnosis node.
    tree.set_diagnosis_nodes(['child1_id'])

    # `child2_id` is forced as diagnosis node and then is the only diagnosis appearing in the constraints list
    conditions_df = tree.compute_conditions_df(min_diagnosis_images=1,
                                               force_diagnosis_ids=['child2_id'],
                                               constrain_diagnosis_ids=['child2_id'])
    actual_diagnosis_nodes = sorted(conditions_df['diagnosis_id'].tolist())
    actual_conditions_nodes = sorted(conditions_df['condition_id'].tolist())
    assert actual_diagnosis_nodes == ['child2_id'], actual_diagnosis_nodes
    assert actual_conditions_nodes == ['child2_id'], actual_conditions_nodes

    # Due to all the constraints imposed, no diagnosis will be selected, `conditions_df` will be None
    conditions_df = tree.compute_conditions_df(min_diagnosis_images=2,
                                               force_diagnosis_ids=['child2_id'],
                                               constrain_diagnosis_ids=['child1_id'])
    assert conditions_df is None


def test_get_ancestors_diagnosis_ids_map(ontology_tree):
    expected = {'AIP:root': ['AIP:0002471', 'AIP:0002491'],
                'AIP:0000000': ['AIP:0002471', 'AIP:0002491'],
                'AIP:0002475': ['AIP:0002491'],
                'AIP:0100001': ['AIP:0002491'],
                'AIP:0000007': ['AIP:0002471', 'AIP:0002491'],
                'AIP:0000001': ['AIP:0002471', 'AIP:0002491']}
    actual = ontology_tree.get_ancestors_diagnosis_ids_map()
    assert expected == actual
