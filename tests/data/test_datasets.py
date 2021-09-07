import os


from studio.data.datasets import Dataset
from studio.data.modelmap import ModelMap
from backports.tempfile import TemporaryDirectory
from studio.data.review_algorithms import reject_outliers


def test_process_training_data(snapshot_manifest, dummy_ontology, ontology_tree, snapshot_reviews_manifest):
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=0))
    with TemporaryDirectory() as dataset_folder:
        dataset = Dataset(dataset_folder,
                          'test',
                          dummy_ontology,
                          model_map
                          )
        train_df = dataset.process_training_data(snapshot_manifest, reject_outliers)

        assert len(train_df) == 5
        assert os.path.isfile(os.path.join(dataset_folder, 'test', 'ontology.json'))
        assert os.path.isfile(os.path.join(dataset_folder, 'test', 'conditions_df.json'))
        assert os.path.isfile(os.path.join(dataset_folder, 'test', 'diagnosis_df.json'))
        assert os.path.isfile(os.path.join(dataset_folder, 'test', 'training_data_processed.json'))

        train_df = dataset.process_training_data(snapshot_reviews_manifest, reject_outliers, image_type='macroscopic')
        assert len(train_df) == 2


def test_perform_dataset_split(snapshot_manifest, dummy_ontology, ontology_tree):
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=0))
    with TemporaryDirectory() as dataset_folder:
        dataset = Dataset(dataset_folder,
                          'test',
                          dummy_ontology,
                          model_map
                          )
        dataset_df = dataset.process_training_data(snapshot_manifest, reject_outliers)

        train_df, val_df = dataset.perform_dataset_split(dataset_df,
                                                         experiment_name='test_experiment',
                                                         mode='class_fraction',
                                                         split_ratio=0.2,
                                                         split_class_count=None,
                                                         )
        assert len(train_df) == 4
        assert len(val_df) == 1
        assert os.path.isfile(os.path.join(dataset_folder, 'test', 'test_experiment', 'training.json'))
        assert os.path.isfile(os.path.join(dataset_folder, 'test', 'test_experiment', 'validation.json'))


def test_create_data_partition(snapshot_manifest, dummy_ontology, ontology_tree, snapshot_reviews_manifest):
    model_map = ModelMap(ontology_tree.compute_conditions_df(min_diagnosis_images=0))
    with TemporaryDirectory() as dataset_folder:
        dataset = Dataset(dataset_folder,
                          'test',
                          dummy_ontology,
                          model_map
                          )

        dataset_df = dataset.create_data_partition(snapshot_manifest,
                                                   reject_outliers,
                                                   partition_name=None,
                                                   min_reviews=1,
                                                   uncertainty_as_class=False,
                                                   uncertainty_mode='keep',
                                                   uncertainty_threshold=0.5,
                                                   root_directory='/data/lab/images/files')
        assert len(dataset_df) == 5

        dataset_df = dataset.create_data_partition(snapshot_reviews_manifest,
                                                   reject_outliers,
                                                   partition_name=None,
                                                   image_type='dermoscopic',
                                                   min_reviews=1,
                                                   uncertainty_as_class=False,
                                                   uncertainty_mode='keep',
                                                   uncertainty_threshold=1.0,
                                                   root_directory='/data/lab/images/files')

        assert len(dataset_df) == 1
