import os
import json

from PIL import Image
from anytree import AnyNode, RenderTree
from anytree import AsciiStyle, ContStyle, ContRoundStyle, DoubleStyle


def load_image(image_path):
    return Image.open(image_path)


def apply_crop(image, crop_coordinates):
    """
    The format of the image cropping coordinates should be [x, y, width, height] and PIL cropping format is
    [left, upper, right, lower].
    """
    crop_x, crop_y, crop_w, crop_h = crop_coordinates

    left = crop_x
    upper = crop_y
    right = crop_x + crop_w
    lower = crop_y + crop_h

    coordinates = [left, upper, right, lower]

    return image.crop(coordinates)


def resize_image(image, size, keep_aspect_ratio=True, resize_if_smaller=False):
    width, height = size[0], size[1]
    image_width, image_height = image.size
    if keep_aspect_ratio:
        if image_width < image_height:
            size = (width, int(image_height / image_width * width))
        else:
            size = (int(image_width / image_height * height), height)

    if not resize_if_smaller and (image_width <= size[0] or image_height <= size[1]):
        return image
    else:
        return image.resize(size)


def search_tags(string_list, tags):
    if not isinstance(tags, list):
        tags = [tags]
    tag_list = []

    for tag in tags:
        tag_list += [string for string in string_list if tag in string]

    return tag_list


def anytree(file):
    """
    Create AnyTree's nodes with multiple parent nodes from an edges-like JSON file.
    The multiple parent node functionality consists on duplicating a node everytime this one
    is linked to a node that exists in multiple parts in the tree.
    Related library documentation:
        https://anytree.readthedocs.io/

    Args:
        file: path to an edge-like JSON file.
    Return:
        nodes: dict with node's ID and AnyTree's AnyNode objects.
               e.g. {'root': AnyNode('root'), 'n1', AnyNode('n1')}.
    """
    with open(file) as f:
        js_graph = json.load(f)

    nodes = {}
    nodes[0] = AnyNode(id="root")

    idx = 1
    for edge in js_graph['edges']:
        parents = [node for node in nodes.values() if node.id == edge['from']]
        for parent in parents:
            nodes[idx] = AnyNode(id=edge['to'], parent=parent)
            idx += 1
    return nodes


def showAnytree(nodes, root_id='root', style='contround', filename=None):
    """
    Render AnyTree's list of nodes.
    Related library documentation:
        https://anytree.readthedocs.io/en/latest/api/anytree.render.html

    Args:
        nodes: dict with node's ID and AnyTree's Node object.
               e.g. {'root': Node('root'), 'n1', Node('n1')}.
        root_id: root node's identifier.
        style: a string indicating the type of line used to represent the parent/child relationship.
               Supported: 'ascii', 'cont', 'contround', 'double'.
        filename: (optional) a string that indicates the location of where to save the TXT file.
                If `filename=None`, then display to screen.
                Else, save to disk at location `filename`.
    """
    if style == 'ascii':
        style = AsciiStyle()
    elif style == 'cont':
        style = ContStyle()
    elif style == 'contround':
        style = ContRoundStyle()
    elif style == 'double':
        style = DoubleStyle()
    else:
        raise ValueError('Style requested not valid.')

    root = [node for node in nodes.values() if node.id == root_id][0]
    if filename:
        # If the filename currently exists, remove it. Otherwise the graph's content will be appended.
        try:
            os.remove(filename)
        except OSError:
            pass

        with open(filename, 'a') as f:
            for pre, _, node in RenderTree(root, style):
                f.write("%s%s\n" % (pre, node.id))
    else:
        for pre, _, node in RenderTree(root, style):
            print("%s%s" % (pre, node.id))
