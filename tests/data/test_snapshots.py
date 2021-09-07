from PIL import Image
from studio.data.snapshots import Snapshot
from backports.tempfile import TemporaryDirectory


def test_compute_node_frequency(snapshot, snapshot_with_reviews):
    node_frequency_df = snapshot.compute_node_frequency()

    assert node_frequency_df.iloc[0, 0] == 'AIP:0002491'
    assert node_frequency_df.iloc[0, 1] == 2
    assert node_frequency_df.iloc[1, 0] == 'AIP:0002478'
    assert node_frequency_df.iloc[1, 1] == 2
    assert node_frequency_df.iloc[2, 0] == 'AIP:0002471'
    assert node_frequency_df.iloc[2, 1] == 1
    assert len(node_frequency_df) == 3

    # We have 5 samples
    assert sum(node_frequency_df['frequency']) == 5

    node_frequency_df = snapshot_with_reviews.compute_node_frequency(uncertainty_threshold=0.5)
    assert node_frequency_df.iloc[0, 0] == 'AIP:0002491'
    assert round(node_frequency_df.iloc[0, 1], 2) == 2.41
    assert node_frequency_df.iloc[1, 0] == 'AIP:0002471'
    assert round(node_frequency_df.iloc[1, 1], 2) == 0.97
    assert node_frequency_df.iloc[2, 0] == 'uncertainty'
    assert round(node_frequency_df.iloc[2, 1], 2) == 0.49
    assert node_frequency_df.iloc[3, 0] == 'AIP:0002480'
    assert round(node_frequency_df.iloc[3, 1], 2) == 0.12
    assert len(node_frequency_df) == 4

    # We have 5 samples but drop 1 due to high uncertainty
    assert round(sum(node_frequency_df['frequency'])) == 4

    node_frequency_df = snapshot_with_reviews.compute_node_frequency(uncertainty_threshold=1.1)
    assert node_frequency_df.iloc[0, 0] == 'AIP:0002491'
    assert round(node_frequency_df.iloc[0, 1], 2) == 2.65
    assert node_frequency_df.iloc[1, 0] == 'AIP:0002471'
    assert round(node_frequency_df.iloc[1, 1], 2) == 1.17
    assert node_frequency_df.iloc[2, 0] == 'uncertainty'
    assert round(node_frequency_df.iloc[2, 1], 2) == 1.06
    assert node_frequency_df.iloc[3, 0] == 'AIP:0002480'
    assert round(node_frequency_df.iloc[3, 1], 2) == 0.12
    assert len(node_frequency_df) == 4

    assert round(sum(node_frequency_df['frequency'])) == 5


def test_get_reviews(snapshot, snapshot_with_reviews):
    # No reviews
    expected = []
    assert expected == snapshot.get_reviews('reviewer_1')
    # With Reviews
    expected = [{'AIP:0002471': 60, 'AIP:0002480': 20, 'uncertainty': 20},
                {},
                {'uncertainty': 100}
                ]
    assert expected == snapshot_with_reviews.get_reviews('reviewer_1')


def test_compute_average_reviews(snapshot, snapshot_with_reviews, snapshot_reviews_manifest):
    # Manifest without reviews
    processed_manifest_df = snapshot.compute_average_reviews()

    expected = [{'AIP:0002491': 1.0}, {'AIP:0002491': 1.0}, {'AIP:0002478': 1.0},
                {'AIP:0002478': 1.0}, {'AIP:0002471': 1.0}, {}, {}]

    for i, review in enumerate(processed_manifest_df['average_reviews']):
        assert review == expected[i]

    assert len(processed_manifest_df) == 7

    # Manifest with reviews, keep uncertainty
    processed_manifest_df = snapshot_with_reviews.compute_average_reviews()

    expected = [{'AIP:0002491': 1.0},
                {'AIP:0002491': 1.0},
                {'AIP:0002471': 0.54, 'AIP:0002480': 0.12, 'uncertainty': 0.16, 'AIP:0002491': 0.18},
                {'uncertainty': 0.5666666666666667,
                 'AIP:0002491': 0.23333333333333334, 'AIP:0002471': 0.2},
                {'uncertainty': 0.3333333333333333, 'AIP:0002491': 0.23333333333333334,
                 'AIP:0002471': 0.43333333333333335}]

    for i, review in enumerate(processed_manifest_df['average_reviews']):
        assert review == expected[i]

    assert len(processed_manifest_df) == 5

    # Manifest with reviews, keep uncertainty, bad_quality tag defined
    processed_manifest_df = snapshot_with_reviews.compute_average_reviews(bad_quality_tag='bad_quality')

    expected = [{'AIP:0002491': 1.0},
                {'AIP:0002491': 1.0},
                {'AIP:0002471': 0.54, 'AIP:0002480': 0.12, 'uncertainty': 0.16, 'AIP:0002491': 0.18},
                {'bad_quality': 0.3333333333333333, 'AIP:0002491': 0.23333333333333334, 'AIP:0002471': 0.2,
                 'uncertainty': 0.23333333333333334},
                {'uncertainty': 0.3333333333333333, 'AIP:0002491': 0.23333333333333334,
                 'AIP:0002471': 0.43333333333333335}]
    for i, review in enumerate(processed_manifest_df['average_reviews']):
        assert review == expected[i]
    assert len(processed_manifest_df) == 5

    # Manifest with reviews, drop uncertainty
    uncertainty_mode = 'drop'
    snapshot = Snapshot(snapshot_manifest=snapshot_reviews_manifest)

    processed_manifest_df = snapshot.compute_average_reviews(uncertainty_mode=uncertainty_mode)

    expected = [{'AIP:0002491': 1.0},
                {'AIP:0002491': 1.0},
                {'AIP:0002471': 0.6428571428571429, 'AIP:0002480': 0.14285714285714285,
                 'AIP:0002491': 0.21428571428571427},
                {'AIP:0002491': 0.5384615384615384, 'AIP:0002471': 0.46153846153846156},
                {'AIP:0002491': 0.35, 'AIP:0002471': 0.65}]
    for i, review in enumerate(processed_manifest_df['average_reviews']):
        assert review == expected[i]
    assert len(processed_manifest_df) == 5

    # Manifest with reviews, distribute uncertainty
    uncertainty_mode = 'distribute'
    snapshot = Snapshot(snapshot_manifest=snapshot_reviews_manifest)

    processed_manifest_df = snapshot.compute_average_reviews(uncertainty_mode=uncertainty_mode)

    expected = [{'AIP:0002491': 1.0},
                {'AIP:0002491': 1.0},
                {'AIP:0002471': 0.68, 'AIP:0002480': 0.14, 'AIP:0002491': 0.18},
                {'AIP:0002491': 0.7, 'AIP:0002471': 0.3},
                {'AIP:0002491': 0.35, 'AIP:0002471': 0.65}]
    for i, review in enumerate(processed_manifest_df['average_reviews']):
        assert review == expected[i]
    assert len(processed_manifest_df) == 5


def test_resize_and_save(animals_catdog_val_manifest, animals_catdog_val_path):
    snapshot = Snapshot(animals_catdog_val_manifest, root_directory=animals_catdog_val_path)
    with TemporaryDirectory() as output_test_dir:
        # Keep aspect ratio False, resize_if_smaller True -- All images will have size of (300, 300)
        df = snapshot.resize_and_save(output_test_dir,
                                      size=(300, 300),
                                      keep_aspect_ratio=False,
                                      apply_crop=False,
                                      resize_if_smaller=True,
                                      column_id='filename')
        for file in df['filename']:
            assert Image.open(file).size == (300, 300)

        # Keep aspect ratio False, Cropping True -- All have size of (250, 300)
        df = snapshot.resize_and_save(output_test_dir,
                                      size=(250, 300),
                                      keep_aspect_ratio=False,
                                      apply_crop=True,
                                      resize_if_smaller=True,
                                      column_id='filename')
        for file in df['filename']:
            assert Image.open(file).size == (250, 300)

        # Keep aspect ratio True, Cropping True
        df = snapshot.resize_and_save(output_test_dir,
                                      size=(300, 300),
                                      keep_aspect_ratio=True,
                                      apply_crop=True,
                                      resize_if_smaller=False,
                                      column_id='filename')

        # All the images are bigger than 300 except for the one cropped which is not resized
        for file in df['filename'][1:]:
            assert max(Image.open(file).size) >= 300

        # First image has crop coordinates so the image would be cropped to 150, 160 as resize if smaller is False
        assert Image.open(df['filename'][0]).size == (150, 160)

        # Keep aspect ratio True, Cropping True, Resize if smaller True, all images should have a minimum size
        # corresponding with minimum of their original dimension (e.g. if width < height --> width size would be 300)
        # Original = (400, 500) ==> cropped = (300, 375)

        df = snapshot.resize_and_save(output_test_dir,
                                      size=(300, 320),
                                      keep_aspect_ratio=True,
                                      apply_crop=True,
                                      resize_if_smaller=True,
                                      column_id='filename')
        for file in df['filename'][0:]:
            size = Image.open(file).size
            if size[0] < size[1]:
                assert size[0] == 300
            else:
                assert size[1] == 320


def test_exclude_reviewers(snapshot_reviews_manifest):
    snapshot = Snapshot(snapshot_manifest=snapshot_reviews_manifest)
    snapshot.exclude_reviewers("reviewer_1")
    assert len(snapshot.manifest_df['reviews'][2]) == 4
    assert 'reviewer_1' not in snapshot.manifest_df['reviews'][2]
    snapshot.exclude_reviewers(["reviewer_2", "reviewer_3"])
    assert len(snapshot.manifest_df['reviews'][2]) == 2
    assert 'reviewer_2' not in snapshot.manifest_df['reviews'][2]
    assert 'reviewer_3' not in snapshot.manifest_df['reviews'][2]


def test_add_labels_source_and_type(snapshot_reviews_manifest):
    snapshot = Snapshot(snapshot_manifest=snapshot_reviews_manifest)
    # There is a tie in the third position so expected[2] will be None
    expected_type = ['macroscopic', None, None, 'dermoscopic', 'macroscopic']
    expected_source = ['AIP', 'AIP', 'review_pipeline', 'review_pipeline', 'review_pipeline']
    image_types = snapshot.manifest_df['image_type'].tolist()
    image_source = snapshot.manifest_df['labels_source'].tolist()
    assert image_types == expected_type
    assert image_source == expected_source
    assert snapshot.rows_tie_image_type == [2]
    assert snapshot.rows_missing_image_type == [1]
