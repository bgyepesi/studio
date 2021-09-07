import studio.data.utils as utils


def test_apply_crop(test_animals_image_path):
    image = utils.load_image(test_animals_image_path)
    actual_image_size = image.size
    expected_image_size = (500, 381)
    assert actual_image_size == expected_image_size

    crop_coordinates = [0, 100, 20, 30]
    image = utils.apply_crop(image, crop_coordinates)
    actual_image_size = image.size
    expected_image_size = (20, 30)
    assert actual_image_size == expected_image_size


def test_search_tags():
    string_list = [
        "AIP:0002478",
        "name:acne-fulminans",
        "AIP:challa"
    ]

    tags = utils.search_tags(string_list, 'AIP:')
    assert tags[0] == "AIP:0002478"
    assert tags[1] == "AIP:challa"

    tags = utils.search_tags(string_list, 'name:')
    assert tags == ["name:acne-fulminans"]
