import os
import json
import pandas as pd

from treelib import Tree
from pandas import json_normalize


class Concept(object):
    def __init__(self, images=None, aggregated_images=None, node_type=None, malignancy=None, display_name=None,
                 node_id=None, node_label=None, show_during_review=None):
        """
        Groups attributes common to a single node in the Ontology.
        Args:
            images: (integer) Number of images from this specific node and its descendants
            aggregated_images: (integer) Number of images assigned to this node and all of its ancestors.
            node_type: (string) Type of node (e.g. general, diagnosis, etc.)
            malignancy: (string) Node concept malignancy (e.g. benign, malignant)
            display_name: (string) A custom string used when displaying the tree.
            show_during_review: (boolean) If True will show the nodes on review tasks
        """
        self.images = images
        self.aggregated_images = aggregated_images
        self.type = node_type
        self.malignancy = malignancy
        self.display_name = display_name
        self.id = node_id
        self.label = node_label
        self.show_during_review = show_during_review

    def __getitem__(self, x):
        """ Allows for a subscriptable class."""
        return getattr(self, x)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class Ontology(Tree):
    # String indicating if a node is a diagnosis node.
    NODE_TYPE_DIAGNOSIS = 'diagnosis'
    NODE_TYPE_GENERAL = 'general'

    # Strings indicating the malignancy of a node.
    # Not specified indicates the node is either composed of sub-nodes that are both malignant and benign,
    # or that this node is not yet reviewed.
    MALIGNANCY_UNSPECIFIED = 'unspecified'
    MALIGNANCY_BENIGN = 'benign'  # Not cancerous.
    MALIGNANCY_PREMALIGNANT = 'premalignant'  # Benign, but is suspicious and perhaps should be monitored.
    MALIGNANCY_MALIGNANT = 'malignant'  # Cancerous.

    # Common column names and JSON keys.
    NODES = 'nodes'
    EDGES = 'edges'

    NODE_ID = 'id'  # Typically this will be the Aip ID of the disease.
    NODE_LABEL = 'label'  # Human readable name.
    NODE_IMAGES = 'images'
    NODE_AGG_IMAGES = 'aggregated_images'
    NODE_TYPE = 'type'
    NODE_MALIGNANCY = 'malignancy'
    NODE_PARENT_ID = 'parent_id'
    NODE_SHOW_DURING_REVIEW = 'show_during_review'

    FROM_NODE_ID = 'from'
    TO_NODE_ID = 'to'

    def __init__(self, ontology_manifest, root_id='AIP:root'):
        """
        Create an Ontology tree.
        Args:
            ontology_manifest: JSON file that defines the ontology as `nodes` and `edges`.
                `ontology_manifest` can be:
                - a string indicating the filename and path to the JSON file.
                - or the already loaded JSON file.

                Assumes the `nodes` information include: `id`, `label`, `node_type`, and `malignancy`.
                Assumes the `edges` will contain (at least) one node pointing to `root`.

            root_id: A string indicating the ID of the root node.

        Returns: Nothing. Stores the following self variables:
            self: Treelib object.
        """
        # Keep track of the root_id.
        self.root_id = root_id

        if isinstance(ontology_manifest, str):
            # This is a string that points to a JSON file.
            with open(ontology_manifest) as f:
                ontology_manifest = json.load(f)

        Tree.__init__(self)
        self._build(root_id, ontology_manifest)
        # Dataframe to keep track of errors
        self.errors_report = None
        self.perform_error_check()

        if len(self.errors_report) > 0:
            print('There have been errors during the Ontology creation, you can see them by accessing to '
                  '`Ontology.errors_report`')
            print(self.errors_report)
        else:
            print('Ontology was created without errors.')

    def _build(self, root_id, ontology_manifest):
        """Build the Ontology tree based on the `ontology_manifest`."""

        nodes_df = json_normalize(ontology_manifest[self.NODES])
        edges_df = json_normalize(ontology_manifest[self.EDGES])

        # Create `root` node
        root = nodes_df.loc[nodes_df[Ontology.NODE_ID] == root_id].iloc[0]
        data = Concept(
            aggregated_images=0,
            node_type=root[Ontology.NODE_TYPE],
            malignancy=root[Ontology.NODE_MALIGNANCY],
            node_id=root[Ontology.NODE_ID],
            node_label=root[Ontology.NODE_LABEL],
            show_during_review=bool(root[Ontology.NODE_SHOW_DURING_REVIEW])
        )
        self.create_node(tag=root[Ontology.NODE_LABEL], identifier=root[Ontology.NODE_ID], data=data)

        # Define `nodes` pointing all to `root` node.
        for index, node in nodes_df.iterrows():
            if node.id != root.id:
                data = Concept(
                    aggregated_images=0,
                    node_type=node[Ontology.NODE_TYPE],
                    malignancy=node[Ontology.NODE_MALIGNANCY],
                    node_id=node[Ontology.NODE_ID],
                    node_label=node[Ontology.NODE_LABEL],
                    show_during_review=node[Ontology.NODE_SHOW_DURING_REVIEW]
                )

                self.create_node(tag=node.label,
                                 identifier=node.id,
                                 parent=root.id,
                                 data=data)

        # Define "edges" respectively
        for index, edge in edges_df.iterrows():
            # If the edge points to a node ID that does not exist, throw an error with the missing node ID.
            if edge.to not in self.nodes:
                raise ValueError("Error: `edge.to={}` node id not found.".format(edge.to))
            if edge['from'] not in self.nodes:
                raise ValueError("Error: `edge.from={}` node id not found.".format(edge['from']))

            self.move_node(source=edge.to, destination=edge['from'])

    def perform_error_check(self):
        errors_report = []
        errors = self._check_label_uniqueness()
        if len(errors) > 0:
            errors_report.append({'Error type': 'Duplicate Labels', 'Nodes': [], 'Collisions': errors})
            print('There nodes with duplicate labels')

        errors = self._check_diagnosis_nodes_collisions_with_descendants()
        if len(errors) > 0:
            for key, value in errors.items():
                errors_report.append({'Error type': 'DN Uniqueness', 'Nodes': key, 'Collisions': errors[key]})
            print('There are diagnosis nodes with at least one diagnosis node in the same branch')

        errors = self._check_same_malignancy_in_descendants()
        if len(errors) > 0:
            for key, value in errors.items():
                errors_report.append({'Error type': 'Different Malignancy', 'Nodes': key, 'Collisions': errors[key]})
            print('There are nodes with at least one different malignancy type in the same branch')

        self.errors_report = pd.DataFrame(errors_report)

    @staticmethod
    def get_node_summary(node, count_images=False):
        """Return a descriptive string that summarizes the ontology node.

        Args:
            node: represents a node in the ontology.
                Assumes the existence of `node` properties (e.g., `node['id']` and `node['label']`).
            count_images: If True, include the image count in the display.
        """
        # Assumes these properties exist.
        node_id = node[Ontology.NODE_ID]
        node_label = node[Ontology.NODE_LABEL]
        node_type = node[Ontology.NODE_TYPE]
        node_malignancy = node[Ontology.NODE_MALIGNANCY]

        if node_type == Ontology.NODE_TYPE_DIAGNOSIS:
            diagnosis_symbol = '*'
        else:
            diagnosis_symbol = ''

        if node_malignancy == Ontology.MALIGNANCY_BENIGN:
            malignancy_symbol = '(B)'
        elif node_malignancy == Ontology.MALIGNANCY_MALIGNANT:
            malignancy_symbol = '(M)'
        elif node_malignancy == Ontology.MALIGNANCY_PREMALIGNANT:
            malignancy_symbol = '(P)'
        else:
            malignancy_symbol = ''

        # Show the name of the node, the Aip ID, malignancy symbol, and a diagnosis node symbol.
        display_name = node_label + " [" + node_id + "]" + malignancy_symbol + diagnosis_symbol

        if count_images:
            n_images = node[Ontology.NODE_IMAGES]
            n_agg_images = node[Ontology.NODE_AGG_IMAGES]
            image_str = "[" + str(n_images) + "]" + "[" + str(n_agg_images) + "A]"
            display_name += image_str

        return display_name

    def get_node_id2name(self, node_list=None):
        node_id2name = {}
        if node_list is None:
            node_list = self.all_nodes()
        for node in node_list:
            node_id2name[node.data['id']] = node.data['label']
        return node_id2name

    def get_ontology_diagnosis_partition(self):
        diagnosis_ids = self.get_diagnosis_ids()
        if len(diagnosis_ids) < 1:
            raise Exception('To call this function you must set diagnosis nodes first.')
        ancestors_dn = set()
        dn_or_children = set()
        for node_id in diagnosis_ids:
            dn_or_children.update(set(self.get_descendants(node_id)))
            s = set(self.get_ancestors(node_id))
            s.remove(node_id)
            ancestors_dn.update(s)

        outside_dn = set(self.nodes).difference(ancestors_dn | dn_or_children)

        return ancestors_dn, dn_or_children, outside_dn

    def display(self, filename=None, line_type='ascii-emv', count_images=False):
        """Display the structure of the ontology tree.

        If `filename` is provided, save to disk at location `filename`.
        Else if `filename=None`, then display to screen.

        Args:
            filename: (optional) a string that indicates the location of where to save the CSV file.
                If `filename=None`, then display to screen.
                Else, save to disk at location `filename`.
            line_type: a string indicating the type of line used to represent the parent/child relationship.
            count_images: If True, include the image count in the display.
        """

        # The name of the field belonging to `node.data` to display.
        data_property = 'display_name'

        for node in self.all_nodes():
            node.data.display_name = self.get_node_summary(node.data, count_images)

        if filename:
            # If the filename currently exists, remove it. Otherwise the ontology will be appended.
            try:
                os.remove(filename)
            except OSError:
                pass

            # Note the `key=self.get_node_id`.
            # This sorts the tree by the node identifier rather than the tag.
            # This is done since the node.tag may change, but the node.identifier likely will not.
            # This makes it easier to compare two ontologies line by line and visualize the changes.
            self.save2file(filename=filename, data_property=data_property, line_type=line_type, key=self.get_node_id)
        else:
            self.show(data_property=data_property, line_type=line_type, key=self.get_node_id)

    def set_diagnosis_nodes(self, node_ids):
        """Initialize all node types to `general`, set `node_ids` types to `diagnosis` and return missing `node_ids`"""
        # Set all the nodes to be of type `general`.
        for node in self.all_nodes():
            node.data[Ontology.NODE_TYPE] = Ontology.NODE_TYPE_GENERAL

        # Track any IDs that do not occur in the ontology.
        missing_node_ids = []

        # Assign the diagnosis `node_type` to the given diagnosis IDs.
        for node_id in node_ids:
            if self.contains(node_id):
                self.nodes[node_id].data[Ontology.NODE_TYPE] = Ontology.NODE_TYPE_DIAGNOSIS
            else:
                print("Warning: Missing node ID=`{}` is missing from the ontology.".format(node_id))
                missing_node_ids.append(node_id)

        # Perform error check including diagnosis nodes uniqueness check.
        self.perform_error_check()

        return missing_node_ids

    def diagnosis_dataframe(self, order_by='tree-level'):
        """Return a dataframe with the diagnosis nodes' IDs and labels."""
        df = self.to_dataframe(order_by=order_by)
        return df[df[Ontology.NODE_TYPE] == Ontology.NODE_TYPE_DIAGNOSIS][[Ontology.NODE_ID, Ontology.NODE_LABEL]]

    def get_name(self, node_id):
        """Returns the node's label name for a given id."""
        if self.contains(node_id):
            return self.get_node(node_id).tag
        else:
            raise ValueError('The node id does not exist.')

    def get_id(self, node_name):
        """Returns the node's identifier for a given name."""
        for node in self.all_nodes():
            if node.tag == node_name:
                return node.identifier
        raise ValueError('The node name does not exist.')

    def get_images(self, node_id):
        """Returns the node's images for a given id."""
        if self.contains(node_id):
            if self.get_node(node_id).data.images is not None:
                return self.get_node(node_id).data.images
            else:
                raise Exception('Image count has not been set, please run set_node_count()')
        else:
            raise ValueError('The `node_id={}` does not exist.'.format(node_id))

    def get_aggregated_images(self, node_id):
        """Returns the node's images for a given id."""
        if self.contains(node_id):
            if self.get_node(node_id).data.aggregated_images is not None:
                return self.get_node(node_id).data.aggregated_images
            else:
                raise Exception('Image count has not been set, please run set_node_count()')
        else:
            raise ValueError('The `node_id={}` does not exist.'.format(node_id))

    def get_malignancy(self, node_id):
        """Returns the node's malignancy for a given id."""
        if self.contains(node_id):
            return self.get_node(node_id).data.malignancy
        else:
            raise ValueError('The node id does not exist.')

    def sort_by(self, nodes, mode='level'):
        """
        Return a sorted list of `nodes` by `mode`.
        Args:
            nodes: list of node identifiers.
            mode: sorting mode. Currently supported:
                    - `level`: nodes sorted by ascending tree level.
        Return:
            sorted_nodes: list of nodes sorted by mode.
        """
        modes = ['level']
        if mode not in modes:
            raise ValueError("Error: invalid option for `sorting`={}".format(mode))

        sorted_nodes = []
        if mode == 'level':
            for level in range(self.depth() + 1):
                for node in nodes:
                    if self.level(node) == level:
                        sorted_nodes.append(node)

        return sorted_nodes

    def get_internal_nodes(self):
        """
        Return a list of all internal (also known as inner) nodes in the tree.
        Returns:
            internal_nodes: list of all internal nodes in the tree.
        """
        internal_nodes = []
        for node in self.all_nodes():
            if self.children(node.identifier):
                internal_nodes.append(node)

        return internal_nodes

    def remove_nodes(self, node_ids):
        """
        Remove the nodes from the tree that are in the list of `node_ids`.
        Will silently fail if the node_id does not exist.
        Args:
            node_ids: list of node identifiers to remove.
        Returns: The number of nodes removed.
        """
        n_removed = 0
        for node_id in node_ids:
            if self.contains(node_id):
                n_removed += self.remove_node(node_id)

        return n_removed

    def connected_node_ids(self, target_ids):
        """
        Return the node IDs for all nodes that connect the `target_ids` nodes to the root node.
        Args:
            target_ids: A list of node IDs to search for.
        Returns: The unique node IDs that connect the target nodes to the root (also included).
        """
        connected_node_ids = []
        for target_id in target_ids:
            nodes_to_root = self.rsearch(target_id)
            for nid in nodes_to_root:
                connected_node_ids.append(nid)

        return list(set(connected_node_ids))

    def remove_unconnected_nodes(self, keep_node_ids):
        """
        Remove from the tree all nodes that are not connected to those the nodes in `keep_nodes_ids`.
        `Connected` is defined in self.connected_node_ids().
        Args:
            keep_node_ids: A list of node IDs indicating those nodes to keep within the tree.
        Returns: The number of nodes removed.
        """
        node_ids = [node_id for node_id in self.nodes]
        nodes_to_preserve = self.connected_node_ids(keep_node_ids)
        nodes_to_remove = list(set(node_ids) - set(nodes_to_preserve))
        n_removed = self.remove_nodes(nodes_to_remove)
        return n_removed

    def not_leaf_nodes(self):
        """Get all nodes that are not leaf nodes."""
        not_leafs = []
        for node in self.all_nodes():
            if not node.is_leaf():
                not_leafs.append(node)

        return not_leafs

    def nodes_to_root(self, node_id):
        """Return the path of nodes from the `node_id` to the root node (root's id inclusive)."""
        nodes = []
        node = self.get_node(node_id)
        while node is not None:
            nodes.append(node)
            node = self.parent(node.identifier)

        return nodes

    def get_ancestors(self, node_id):
        """Return the ancestors of `node_id`from `node_id` to the root node (root's id inclusive)."""
        nodes = []
        node = self.get_node(node_id)
        while node is not None:
            nodes.append(node.identifier)
            node = self.parent(node.identifier)

        return nodes

    def get_descendants(self, node_id):
        """
        Return the descendant nodes of `node_id`
        Args:
            node_id: node identifier
        Returns:
            descendant_ids: list of node identifiers (including `node_id`)
        """
        descendant_ids = [descendant_id for descendant_id in self.subtree(node_id).nodes]
        return descendant_ids

    def get_diagnosis_ids(self, node_id=None):
        """
        Return diagnosis node identifiers from the subtree specified by the node_id. By default will return all of them.
        Args:
            node_id: node identifier. The default is the tree root `self.root_id`
        Returns:
            diagnosis_node_ids: list of node identifiers labeled as `diagnosis` type in the ontology.
        """
        node_id = node_id or self.root_id
        node_ids = self.get_descendants(node_id)
        diagnosis_node_ids = self.node_ids_of_type(node_ids, target_type=self.NODE_TYPE_DIAGNOSIS)
        return diagnosis_node_ids

    def get_ancestors_diagnosis_ids_map(self):
        ancestors_diagnosis_nodes_map = {}
        ancestors_set, _, _ = self.get_ontology_diagnosis_partition()
        for ancestor in ancestors_set:
            ancestors_diagnosis_nodes_map[ancestor] = self.get_diagnosis_ids(ancestor)
        return ancestors_diagnosis_nodes_map

    def branch_node_ids(self, node_id):
        """
        Return the node IDs for all those nodes within the same branch.
        Args:
            node_id: node identifier
        Returns:
            branch_ids: list of node identifiers for all those nodes within the same branch as `node_id`.
        """
        ancestor_ids = [ancestor_id for ancestor_id in self.rsearch(nid=node_id)]
        descendant_ids = self.get_descendants(node_id)

        # All the node_ids in a branch. Include myself.
        branch_ids = list(set(ancestor_ids + descendant_ids))
        return branch_ids

    def node_ids_of_type(self, node_ids, target_type='diagnosis'):
        """
        Return the nodes contained in `node_ids` that fulfill the `target_type`.
        Args:
            node_ids: list of node identifiers
        Returns:
            target_node_ids: list of node identifiers filtered from `node_ids` that fulfills the `target_type`
        """
        target_node_ids = []
        for node_id in node_ids:
            node_type = self.nodes[node_id].data[Ontology.NODE_TYPE]
            if node_type == target_type:
                target_node_ids.append(node_id)

        return target_node_ids

    def nodes_coverage_by_diagnosis(self):
        """
        Return the ontology nodes covered and not covered by diagnosis nodes.
        Returns:
            covered_ids: list of node identifiers that fall in any diagnosis node's branch
            not_covered_ids: list of node identifiers that don't fall in any diagnosis node's branch
        """
        covered_ids = []
        not_covered_ids = []

        for node_id in self.nodes:
            branch_ids = self.branch_node_ids(node_id)
            branch_diagnosis_ids = self.node_ids_of_type(branch_ids, target_type='diagnosis')

            if len(branch_diagnosis_ids) == 0:
                # This branch does not have a diagnosis node.
                # This node should be removed.
                not_covered_ids.append(node_id)
            else:
                # This branch has diagnosis nodes >= 1.
                # Possible to have >1 since includes the sub-tree.
                covered_ids.append(node_id)

        return covered_ids, not_covered_ids

    def _check_label_uniqueness(self):
        """
        Traverse tree and search for duplicate labels
        """
        node_ids = [node_id for node_id in self.nodes]
        duplicates, seen = list(), set()
        for node_id in node_ids:
            label = self.nodes[node_id].data[Ontology.NODE_LABEL]
            if label in seen:
                duplicates.append(label)
            else:
                seen.add(label)
        return duplicates

    def _check_diagnosis_nodes_collisions_with_descendants(self):
        """
        Collisions are defined as the appearance of diagnosis node(s) in a diagnosis node descendants branch
        """
        errors = {}
        for node_id in self.nodes:
            if self.nodes[node_id].data[Ontology.NODE_TYPE] == Ontology.NODE_TYPE_DIAGNOSIS:
                node_collisions = self.node_ids_of_type(self.get_descendants(node_id), target_type='diagnosis')
                # > 1 because descendants include node id
                if len(node_collisions) > 1:
                    errors[node_id] = node_collisions[1:]
        return errors

    def _check_same_malignancy_in_descendants(self):
        """
        Malignancy collision is defined as the descendant node(s) that have been specified and mismatch
        the specified malignancy of an ancestor
        """
        errors = {}
        for node_id in self.nodes:
            node_colisions = []
            malignancy = self.nodes[node_id].data[Ontology.NODE_MALIGNANCY]
            if malignancy != Ontology.MALIGNANCY_UNSPECIFIED:
                for descendant_id in self.get_descendants(node_id):
                    malignancy_descendant = self.nodes[descendant_id].data[Ontology.NODE_MALIGNANCY]
                    if malignancy_descendant != Ontology.MALIGNANCY_UNSPECIFIED and malignancy_descendant != malignancy:
                        node_colisions.append(descendant_id)
                if len(node_colisions) > 0:
                    errors[node_id] = node_colisions
        return errors

    def _reset_node_count(self, mode='images'):
        """
        Set `images` or `aggregated_images` to zero.
        Args:
            mode: either `images` or `aggregated_images`.
                - `images` set node.data.images to zero.
                - `aggregated_images` set node.data.aggregated_images to zero.
        Returns:
            Nothing. Set node's image count based on `mode` to zero.
        """
        supported_modes = ['images', 'aggregated_images']
        if mode not in supported_modes:
            raise ValueError('Image reset mode not supported. Only `images` and `aggregated_images` are supported.')

        for node in self.all_nodes():
            if mode == 'images':
                node.data.images = 0
            if mode == 'aggregated_images':
                node.data.aggregated_images = 0

    def set_node_count(self, nodes_frequency):
        """
        Assign the node's `images` and `aggregated_images` attribute according to `nodes_frequency`

        Args:
            nodes_frequency: dataframe containing `node_id` and `frequency` values.

        Returns: Nothing.
            Sets all node's `images` and `aggregated_images` as defined in `nodes_frequency`.

            `self.missing_nodes contains any `node_id`'s that occur in `nodes_frequency`, but not in our tree.
            This likely indicates that the IDs are wrong, or we need to add nodes to our tree.
        """
        self._reset_node_count('images')
        self._reset_node_count('aggregated_images')
        missing_nodes = []  # Keep track of nodes that are not in the tree.

        for index, row in nodes_frequency.iterrows():
            if self.contains(row.node_id):
                self.get_node(row.node_id).data.images = row.frequency
                # self.get_node(row.node_id).data.aggregated_images = row.frequency
            else:
                # Track the node IDs that have images, but we do not currently have in our tree.
                missing_nodes.append(row.node_id)

        self.missing_nodes = missing_nodes

        if len(missing_nodes) > 0:
            print('There are {} `node_id`\'s that occur in `nodes_frequency`, but not in our tree. '
                  'This likely indicates that the IDs are wrong, or we need to add nodes to our tree. This is the list:'
                  '{}'.format(len(missing_nodes), missing_nodes))

        # Compute node.data.aggregated_images values
        self.aggregate_images()

    def aggregate_images(self):
        """
        Aggregate the number of images in all the descendant nodes (from leaf to root).

        Returns: Nothing.
            Updates the `node.data.aggregated_images` values based on the aggregation.
        """
        # Reset and initialize aggregated_images as images count
        self._reset_node_count('aggregated_images')
        for node_id in self.nodes:
            self.nodes[node_id].data.aggregated_images = self.nodes[node_id].data.images

        # Perform aggregation
        level_nodes = self.nodes_by_level()
        levels = sorted(level_nodes.keys())
        levels.reverse()  # To start at the lowest level.

        for level in levels[:-1]:  # Skip level 0 since there is no parent.
            for node_id in level_nodes[level]:
                self.parent(node_id).data.aggregated_images += self.nodes[node_id].data.aggregated_images

    def nodes_by_level(self):
        """Return a dictionary where keys indicate the tree depth, and values are the node IDs"""
        level_nodes = {}
        for level in range(self.depth() + 1):
            level_nodes[level] = []

        # Store all the nodes in a dict by level.
        for node in self.all_nodes():
            level = self.level(node.identifier)
            level_nodes[level].append(node.identifier)

        return level_nodes

    def prune_nodes_by_image(self, node_ids, min_aggregated_images=0):
        """
        Prune `node_ids` from the tree that have less than a specified number of images.
        Args:
            node_ids (list): list of node identifiers.
            min_aggregated_images (int): Minimum number of aggregated images a node must have to remain in the tree.
        Returns:
            remove_count: The number of nodes removed from the tree.
        """
        remove_count = 0
        for node_id in node_ids:
            if self.contains(node_id):
                node = self.nodes[node_id]
                if node.data.aggregated_images <= min_aggregated_images:
                    remove_count += self.remove_node(node.identifier)

        return remove_count

    def filter_min_images(self, node_ids, min_images=0):
        """
        Removes the nodes in the list node_ids with less `images` than `min_images`.
        Args:
            node_ids (list): list of node identifiers.
            min_images (int): Minimum number of images a node must have to remain in the filtered_nodes.
        Returns:
            filtered_nodes: List of node identifiers with greater than or equal to `min_images`.
        """
        filtered_nodes = []
        for node_id in node_ids:
            if self.get_images(node_id) >= min_images:
                filtered_nodes.append(node_id)
        return filtered_nodes

    def filter_min_aggregated_images(self, node_ids, min_aggregated_images=0):
        """
        Removes the nodes in the list node_ids with less `aggregated_images` than `min_images`.
        Args:
            node_ids (list): list of node identifiers.
            min_aggregated_images (int): Minimum number of aggregated images a node must have to remain in the
                                        filtered_nodes.
        Returns:
            filtered_nodes: List of node identifiers with greater than or equal to `min_aggregated_images`.
        """
        filtered_nodes = []
        for node_id in node_ids:
            if self.get_node(node_id).data.aggregated_images >= min_aggregated_images:
                filtered_nodes.append(node_id)
        return filtered_nodes

    def to_dataframe(self, root_parent_id=None, order_by='tree-level'):
        """Return the ontology in a dataframe format.

        To convert from DataFrame back to JSON, see `self.json_ontology_from_dataframe()`.

        Args:
            root_parent_id: Node identifier to be used as root node.
            order_by: string indicating how the nodes should be ordered.

        Returns:
            df: Dataframe with tree's information.
        """
        node_ids = []
        node_labels = []
        node_types = []
        node_parent = []
        node_malignancies = []
        node_show_reviews = []

        all_node_ids = self.get_ordered_node_ids(order_by=order_by)

        for node_id in all_node_ids:
            node = self.nodes[node_id]
            node_ids.append(node.identifier)
            node_labels.append(node.tag)
            node_types.append(node.data[Ontology.NODE_TYPE])
            node_malignancies.append(node.data[Ontology.NODE_MALIGNANCY])
            node_show_reviews.append(node.data[Ontology.NODE_SHOW_DURING_REVIEW])

            parent = self.parent(node_id)
            if parent is None:
                parent_id = root_parent_id
            else:
                parent_id = parent.identifier
            node_parent.append(parent_id)

        data = {
            Ontology.NODE_ID: node_ids,
            Ontology.NODE_LABEL: node_labels,
            Ontology.NODE_PARENT_ID: node_parent,
            Ontology.NODE_TYPE: node_types,
            Ontology.NODE_MALIGNANCY: node_malignancies,
            Ontology.NODE_SHOW_DURING_REVIEW: node_show_reviews,
        }

        df = pd.DataFrame(data)
        # Order the columns in this specific order.
        df = df[[Ontology.NODE_ID, Ontology.NODE_LABEL, Ontology.NODE_PARENT_ID,
                 Ontology.NODE_TYPE, Ontology.NODE_MALIGNANCY, Ontology.NODE_SHOW_DURING_REVIEW]]

        return df

    def to_json(self, file_path=None, indent=1, export_node_type=True, order_by='tree-level'):
        """Render the ontology in a JSON format.

        Args:
            file_path: String indicating the file name to save the JSON file.
            indent: The amount to indent the JSON file.
            export_node_type: If True, export the node type. Else, export the node type as 'general'.
            order_by: String indicating the order to output the nodes.

        Returns:
            If `file_path` is None, return the JSON string.
        """
        nodes_list = []
        edges_list = []

        node_ids = self.get_ordered_node_ids(order_by=order_by)

        for node_id in node_ids:
            node = self.get_node(node_id)
            json_node = Ontology.json_node_from_node_data(node.data)

            if not export_node_type:
                # Set node type as `general` to decouple the diagnosis node selection from the ontology.
                json_node[Ontology.NODE_TYPE] = Ontology.NODE_TYPE_GENERAL

            nodes_list.append(json_node)

            if not node.is_leaf():
                # The sorted() function maintains a consistent ordering of edges.
                for child in sorted(self.children(node.identifier)):
                    edges_list.append({'from': node.identifier, 'to': child.identifier})

        json_ontology = {"nodes": nodes_list, "edges": edges_list}

        if file_path:
            with open(file_path, 'w') as outfile:
                json.dump(json_ontology, outfile, indent=indent, sort_keys=False)
        else:
            return json.dumps(json_ontology, indent=indent, sort_keys=False)

    def to_jstree_json(self, file_path=None):
        """Render the ontology in a JSON format visualizable with the jstree library.
           Every node in the json has the following keys:
               - `id`: the AIP id of the node
               - `parent` : the AIP id of the parent
               - `text`: the name of node to be displayed by the jstree library
               - `data`: this attributes contains the internal properties of the node:
                 - `images`: the number of the images of the node
                 - `agg_images`: the number of the images of the aggregated images
                 - `malignancy`: the malignancy of the node
                 - `type`: the type of the node i.e. general or diagnosis
                 - `label`: the name of the node

        Args:
            file_path: String indicating the file name to save the JSON file.

        Returns:
            If `file_path` is None, return the JSON string.
        """
        nodes_jstree_list = []

        node_ids = self.get_ordered_node_ids(order_by='tree-level')

        for node_id in node_ids:
            node = self.get_node(node_id)
            json_node = Ontology.json_node_from_node_data(node.data)

            # Add the internal properties `images`, `agg_images`, `malignancy`, `show_during_review` and `type` in the
            # `data` attribute
            json_node['data'] = {}
            json_node['data'][Ontology.NODE_IMAGES] = node.data.images
            json_node['data'][Ontology.NODE_AGG_IMAGES] = node.data.aggregated_images
            json_node['data'][Ontology.NODE_TYPE] = node.data.type
            json_node['data'][Ontology.NODE_LABEL] = node.data.label
            json_node['data'][Ontology.NODE_MALIGNANCY] = node.data.malignancy
            json_node['data'][Ontology.NODE_SHOW_DURING_REVIEW] = node.data.show_during_review

            # Remove the internal peroperties added to the `data` attribute from the `json_node`
            json_node.pop(Ontology.NODE_TYPE)
            json_node.pop(Ontology.NODE_MALIGNANCY)
            json_node.pop(Ontology.NODE_LABEL)
            json_node.pop(Ontology.NODE_SHOW_DURING_REVIEW)

            # Add the `parent` property by following the jstree library's convention
            if node.data.id == self.root_id:
                json_node['parent'] = "#"
            else:
                json_node['parent'] = self.get_ancestors(node.data.id)[1]

            # Add the `text` property expected by jstree
            json_node['text'] = node.data.label
            # Add the `open` property to the node
            json_node['state'] = {'opened': 'true'}

            # Check if the `json_node` contains the expected attributes
            if not set([Ontology.NODE_LABEL, Ontology.NODE_IMAGES, Ontology.NODE_AGG_IMAGES, Ontology.NODE_MALIGNANCY,
                        Ontology.NODE_SHOW_DURING_REVIEW, Ontology.NODE_TYPE]) == set(json_node['data'].keys()):
                raise ValueError('One or more of the properties `{}`, `{}`, `{}`, `{}`, `{}` and `{}` of '
                                 'nodes have not been added.'.format(Ontology.NODE_LABEL,
                                                                     Ontology.NODE_IMAGES,
                                                                     Ontology.NODE_AGG_IMAGES,
                                                                     Ontology.NODE_MALIGNANCY,
                                                                     Ontology.NODE_TYPE,
                                                                     Ontology.NODE_SHOW_DURING_REVIEW))
            if not set([Ontology.NODE_ID, 'parent', 'text', 'data', 'state']) == set(json_node.keys()):
                raise ValueError('The expected properties `{}`, `parent`, `text`, '
                                 '`data` and `state` of the node have not been added.'.format(Ontology.NODE_ID))
            nodes_jstree_list.append(json_node)

        if file_path:
            with open(file_path, 'w') as outfile:
                json.dump(nodes_jstree_list, outfile, indent=1, sort_keys=False)
        else:
            return json.dumps(nodes_jstree_list, indent=1, sort_keys=False)

    def get_ordered_node_ids(self, order_by='tree-level'):
        """Return the node IDs in the order specified by `order_by`.

         This function is similar and could be removed if no longer needed.

        `self.sort_by()`

        Args:
            order_by: a string indicating how the node IDs should be ordered. Modes supported: `tree-level`, `identifier`.

        Returns:
            node_ids: list of ordered node IDs

        """

        if order_by == 'tree-level':
            # Sort by tree level and order by the node identifier within a level.
            node_ids = [node_id for node_id in self.expand_tree(key=self.get_node_id)]
        elif order_by == 'identifier':
            # Ignore tree level and sort by the node identifier.
            node_ids = sorted(self.nodes.keys())
        else:
            raise ValueError("Error: invalid `order_by={}` option.".format(order_by))

        return node_ids

    @staticmethod
    def get_node_id(node):
        return node.identifier

    @staticmethod
    def json_node_from_node_data(node_data):
        json_node = {
            Ontology.NODE_ID: node_data[Ontology.NODE_ID],
            Ontology.NODE_LABEL: node_data[Ontology.NODE_LABEL],
            Ontology.NODE_TYPE: node_data[Ontology.NODE_TYPE],
            Ontology.NODE_MALIGNANCY: node_data[Ontology.NODE_MALIGNANCY],
            Ontology.NODE_SHOW_DURING_REVIEW: node_data[Ontology.NODE_SHOW_DURING_REVIEW]
        }
        return json_node

    @staticmethod
    def json_nodes_from_dataframe(df):
        """Return nodes in a JSON format from a pandas dataframe.

        Args:
            df: pandas dataframe where each row represents a node.

        Returns:
            json_nodes: A list of nodes in the expected JSON format.

        """
        json_nodes = []
        for index, row in df.iterrows():
            json_node = Ontology.json_node_from_node_data(row)
            json_nodes.append(json_node)

        return json_nodes

    @staticmethod
    def json_edges_from_dataframe(df):
        """Return edges in a JSON format from a pandas dataframe.

        Args:
            df: pandas dataframe where each row represents a node.

        Returns:
            json_edges: A list of edges in the expected JSON format.

        """
        json_edges = []
        for index, row in df.iterrows():
            parent_id = row[Ontology.NODE_PARENT_ID]
            # Check if the parent_id is present.
            if pd.isna(parent_id) or (parent_id == ""):
                # If not present, then this is the root node and does not have a parent.
                pass
            else:
                json_edge = {Ontology.FROM_NODE_ID: parent_id, Ontology.TO_NODE_ID: row[Ontology.NODE_ID]}
                json_edges.append(json_edge)

        return json_edges

    @staticmethod
    def json_ontology_from_dataframe(df):
        """Return an ontology in the JSON format from a pandas dataframe.

        Args:
            df: pandas dataframe where each row represents a node.

        Returns:
            json_ontology: A dictionary composed of the nodes and edges that make up the ontology.

        """
        json_nodes = Ontology.json_nodes_from_dataframe(df)
        json_edges = Ontology.json_edges_from_dataframe(df)
        json_ontology = {Ontology.NODES: json_nodes, Ontology.EDGES: json_edges}
        return json_ontology

    def compute_conditions_df(self, min_diagnosis_images=0, force_diagnosis_ids=None, constrain_diagnosis_ids=None):
        """Forms training classes.

        There are many ways to do this.
        One way is to group all the sub-conditions we have data for into a single training
        class that reaches a minimum number of images. This outputs a two-layer tree approach.

        Args:
            min_diagnosis_images: minimum number of aggregated images a diagnosis node must have to
                form a training class.
            force_diagnosis_ids: List of diagnosis node IDs that will always be included even if they do not fulfill
                the `min_diagnosis_images` requirement. This can be used to set a higher image threshold for
                new diagnosis nodes while still including the diagnosis IDs used in previous models.
                WARNING: If this option is selected, the ontology will be modified with the selection of such diagnosis.
            constrain_diagnosis_ids: If provided, all the diagnosis node IDs that are not inside this list will be
                excluded. It is applied over the ontology diagnosis IDs and the IDs contained in `force_diagnosis_ids`.
        """
        conditions_df = []

        if force_diagnosis_ids is None:
            force_diagnosis_ids = []

        diagnosis_ids = self.get_diagnosis_ids()
        self.set_diagnosis_nodes(diagnosis_ids + force_diagnosis_ids)

        filtered_diagnosis_ids = self.filter_min_aggregated_images(diagnosis_ids, min_diagnosis_images)

        # Add back any existing diagnosis_ids
        for diag_id in force_diagnosis_ids:
            if diag_id not in filtered_diagnosis_ids:
                filtered_diagnosis_ids.append(diag_id)

        filtered_diagnosis_ids = sorted(filtered_diagnosis_ids)
        columns = ['class_index', 'diagnosis_name', 'diagnosis_id', 'condition_name', 'condition_id', 'malignancy',
                   'n_samples']

        if not filtered_diagnosis_ids:
            raise ValueError('There is not any diagnosis id with the minimum number of (aggregated) images.')

        class_index = -1
        diagnosis_ids_excluded = []
        for diagnosis_id in filtered_diagnosis_ids:
            if constrain_diagnosis_ids is not None and diagnosis_id not in constrain_diagnosis_ids:
                diagnosis_ids_excluded.append(diagnosis_id)
            else:
                class_index += 1
                train_diagnosis_conditions = list()

                # Get all diagnosis id's descendant nodes (including itself, since may have images associated with it)
                for node_id in self.get_descendants(diagnosis_id):
                    train_diagnosis_conditions.append(node_id)

                # Unique conditions.
                train_diagnosis_conditions = sorted(set(train_diagnosis_conditions),
                                                    key=train_diagnosis_conditions.index)

                for condition_id in train_diagnosis_conditions:
                    # diagnosis_id could also be the `condition_id`.
                    conditions_df.append(
                        {'class_index': class_index,
                         'condition_id': condition_id,
                         'condition_name': self.nodes[condition_id].tag,
                         'diagnosis_id': diagnosis_id,
                         'diagnosis_name': self.nodes[diagnosis_id].tag,
                         'n_samples': self.nodes[condition_id].data.images,
                         'malignancy': self.nodes[diagnosis_id].data.malignancy
                         }
                    )
        self.set_diagnosis_nodes(list(set(filtered_diagnosis_ids) - set(diagnosis_ids_excluded)))
        if len(diagnosis_ids_excluded) > 0:
            print('The following %i diagnosis IDs were excluded:\n %s' % (len(diagnosis_ids_excluded),
                                                                          diagnosis_ids_excluded))

        if len(conditions_df) < 1:
            print('All the conditions were excluded')
            return None

        conditions_df = pd.DataFrame(conditions_df)
        conditions_df = conditions_df[columns]

        return conditions_df
