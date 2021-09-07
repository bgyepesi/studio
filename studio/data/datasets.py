import os

from studio.utils.utils import mkdir
from studio.data.ontology import Ontology
from studio.data.modelmap import ModelMap
from studio.data.snapshots import Snapshot
from studio.utils.data_utils import split_dataframe


class Dataset:
    def __init__(self,
                 dataset_folder,
                 dataset_name,
                 ontology_path,
                 model_map):

        self.model_map = model_map
        # Create Ontology and set diagnosis nodes
        self.ontology_path = ontology_path
        self.ontology = Ontology(ontology_path)
        self.ontology.set_diagnosis_nodes(self.model_map.diagnosis_ids)
        self.dataset_folder = dataset_folder
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(self.dataset_folder, self.dataset_name)

    def process_training_data(self,
                              snapshot_manifest_path,
                              mapping_function,
                              image_type=None,
                              processed_name='training_data_processed',
                              min_reviews=1,
                              uncertainty_as_class=False,
                              uncertainty_mode='distribute',
                              uncertainty_threshold=0.5,
                              images_root_directory='/data/old-data/images/'):
        mkdir(self.dataset_dir)

        print('Processing Snapshot')
        # Create Snapshot Object
        self.snapshot = Snapshot(snapshot_manifest=snapshot_manifest_path, root_directory=images_root_directory)
        # Modify filename column joining paths (root + storage_key)
        self.snapshot.process_filename_column()

        print('The number of total samples contained in the snapshot data is {}'.format(len(self.snapshot.manifest_df)))

        if image_type == 'macroscopic':
            self.snapshot.manifest_df = \
                self.snapshot.manifest_df[self.snapshot.manifest_df['image_type'] == 'macroscopic']
        elif image_type == 'dermoscopic':
            self.snapshot.manifest_df = \
                self.snapshot.manifest_df[self.snapshot.manifest_df['image_type'] == 'dermoscopic']

        print('Processing Snapshot - Done!')

        print('The number of total samples contained in the snapshot data after filtering by image type {} is {}'.format
              (image_type, len(self.snapshot.manifest_df)))

        print('\nComputing Node Frequency and setting ontology counts')
        # Compute Node Frequency
        node_frequency = self.snapshot.compute_node_frequency(uncertainty_threshold=0.5, uncertainty_mode='keep')
        self.ontology.set_node_count(node_frequency)

        # Copy a snapshot of the ontology
        self.ontology.to_json(file_path=os.path.join(self.dataset_dir, 'ontology.json'), export_node_type=False)
        print('\nOntology saved in {}'.format(os.path.join(self.dataset_dir, 'ontology.json')))
        conditions_df = self.ontology.compute_conditions_df(min_diagnosis_images=0,
                                                            force_diagnosis_ids=None,
                                                            constrain_diagnosis_ids=self.model_map.diagnosis_ids
                                                            )

        # Save ModelMap with training images information (such as number of samples)
        self.model_map = ModelMap(conditions_df)

        self.save_dataframe(os.path.join(os.path.join(self.dataset_dir, 'conditions_df.json')),
                            self.model_map.conditions_df)
        self.save_dataframe(os.path.join(os.path.join(self.dataset_dir, 'diagnosis_df.json')),
                            self.model_map.diagnosis_df)

        print('\nModelMap contains: \n- {} diagnosis nodes \n- {} conditions with at least a training image \n- {} '
              'total conditions that map to a diagnosis node'.format(
                  len(self.model_map.diagnosis_df),
                  sum(self.model_map.conditions_df['n_samples'] > 0),
                  len(self.model_map.conditions_df)))

        print('Updated Model Map saved in {}'.format(os.path.join(self.dataset_dir, 'conditions_df.json')))

        print('\nProcessing Training Dataset')
        # Map training data to model map classes
        training_df = self.model_map.map_manifest(self.snapshot.manifest_df,
                                                  mapping_function=mapping_function,
                                                  min_reviews=min_reviews,
                                                  uncertainty_as_class=uncertainty_as_class,
                                                  uncertainty_mode=uncertainty_mode,
                                                  uncertainty_threshold=uncertainty_threshold
                                                  )

        self.save_dataframe(os.path.join(self.dataset_dir, processed_name + '.json'), training_df)

        print('The number of total samples contained in the training data is {}'.format(len(training_df)))
        print('Training Dataset saved in {}'.format(os.path.join(self.dataset_dir, processed_name + '.json')))

        return training_df

    def perform_dataset_split(self,
                              manifest_df,
                              experiment_name=None,
                              mode='class_fraction',
                              split_ratio=0.15,
                              split_class_count=None,
                              ):

        # Split dataset into two partitions
        train_df, val_df = split_dataframe(manifest_df,
                                           mode=mode,
                                           split_ratio=split_ratio,
                                           split_class_count=split_class_count)

        print('Training Set Size: {}'.format(len(train_df)))
        print('Validation Set Size: {}'.format(len(val_df)))

        if experiment_name is not None:
            experiment_path = os.path.join(self.dataset_dir, experiment_name)
            mkdir(experiment_path)

            self.save_dataframe((os.path.join(experiment_path, 'training.json')), train_df)
            self.save_dataframe((os.path.join(experiment_path, 'validation.json')), val_df)

            print('Training dataset split saved in {}'.format(os.path.join(experiment_path, 'training.json')))
            print('Validation dataset split saved in {}'.format(os.path.join(experiment_path, 'validation.json')))

        return train_df, val_df

    def create_data_partition(self,
                              snapshot_manifest_path,
                              mapping_function,
                              image_type=None,
                              partition_name=None,
                              min_reviews=1,
                              uncertainty_as_class=False,
                              uncertainty_mode='distribute',
                              uncertainty_threshold=0.5,
                              root_directory='/data/lab/images/files'):

        print('Processing Snapshot')
        # Create Snapshot Object
        self.snapshot = Snapshot(snapshot_manifest=snapshot_manifest_path, root_directory=root_directory)

        # Modify filename column joining paths (root + storage_key)
        self.snapshot.process_filename_column()
        print('The number of total samples contained in the lab snapshot is {}'.format(
            len(self.snapshot.manifest_df)))

        if image_type == 'macroscopic':
            self.snapshot.manifest_df =\
                self.snapshot.manifest_df[self.snapshot.manifest_df['image_type'] == 'macroscopic']
        elif image_type == 'dermoscopic':
            self.snapshot.manifest_df = \
                self.snapshot.manifest_df[self.snapshot.manifest_df['image_type'] == 'dermoscopic']

        print('Processing Snapshot - Done!')

        print('The number of total samples contained in the snapshot data after filtering by image type {} is {}'.format
              (image_type, len(self.snapshot.manifest_df)))

        # Map training data to model map classes
        print('\nProcessing Dataset')
        dataset_df = self.model_map.map_manifest(self.snapshot.manifest_df,
                                                 mapping_function=mapping_function,
                                                 min_reviews=min_reviews,
                                                 uncertainty_as_class=uncertainty_as_class,
                                                 uncertainty_mode=uncertainty_mode,
                                                 uncertainty_threshold=uncertainty_threshold
                                                 )
        print('The number of total samples contained in the dataset is {}'.format(len(dataset_df)))

        if partition_name is not None:
            self.save_dataframe((os.path.join(self.dataset_dir, partition_name + '.json')), dataset_df)
            print('Dataset processed and saved in {}'.format(os.path.join(self.dataset_dir,
                                                                          partition_name + '.json')))

        return dataset_df

    @staticmethod
    def save_dataframe(filepath, dataframe):
        folderpath = '/'.join(filepath.split('/')[:-1])
        mkdir(folderpath)
        dataframe.to_json(filepath, orient='records', indent=4)
