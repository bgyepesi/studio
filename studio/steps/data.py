import os
import pandas as pd

from studio.utils import data_utils, utils
from studio.data.snapshots import Snapshot
from studio.data.modelmap import ModelMap


class Data(object):

    @staticmethod
    def process_directory(data_dir):
        manifest_df = data_utils.directory_to_dataframe(data_dir)
        return manifest_df

    @staticmethod
    def process_modelmap(snapshot_manifest, data_directory, conditions_manifest):
        # Create Snapshot
        snapshot = Snapshot(snapshot_manifest=snapshot_manifest, root_directory=data_directory)
        # Compute Average Reviews or Ground Truth Labels
        snapshot.compute_average_reviews()
        # Modify filename column joining paths (`root` + `storage_key`)
        snapshot_df = snapshot.process_filename_column(column_id='storage_key')

        # Create ModelMap
        conditions_df = pd.read_json(conditions_manifest)
        modelmap = ModelMap(conditions_df)
        # Map snapshot to ModelMap diagnosis
        manifest_df = modelmap.map_manifest(snapshot_df, mapping_mode='reject_outliers')
        return manifest_df

    @staticmethod
    def process_class_snapshot(class_snapshot_manifest, data_directory):
        snapshot = Snapshot(snapshot_manifest=class_snapshot_manifest, root_directory=data_directory)

        # Append the `data_directory` to the `filename` column
        snapshot.process_filename_column()

        # Append the `class_probabilities` column to the manifests
        manifest_df = snapshot.append_class_probabilities()
        return manifest_df


class TrainData(Data):
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = os.path.join(output_dir, 'train')
        utils.mkdir(self.output_dir)

    def process_directories(self):
        # Read `train_dir` and `val_dir` and convert to dataframe
        train_manifest_df = self.process_directory(self.config['directory']['train_dir'])
        val_manifest_df = self.process_directory(self.config['directory']['val_dir'])

        return train_manifest_df, val_manifest_df

    def process_modelmap(self):
        # Read from config file
        data_directory = self.config['modelmap']['data_directory']
        conditions_manifest = self.config['modelmap']['conditions_manifest_path']
        snapshot_manifest_path = self.config['modelmap']['dataset_manifest_path']
        self.config['modelmap']['validation_split']

        snapshot_df = Data.process_modelmap(snapshot_manifest_path, data_directory, conditions_manifest)

        # Split `snapshot_df` into train and val sets
        if self.config['modelmap']['validation_split'].get('class_ratio'):
            split_ratio = self.config['modelmap']['validation_split']['class_ratio']
            train_manifest_df, val_manifest_df = \
                data_utils.split_dataframe(snapshot_df,
                                           mode='class_fraction',
                                           split_ratio=split_ratio)

        if self.config['modelmap']['validation_split'].get('class_count'):
            split_count = self.config['modelmap']['validation_split']['class_count']
            train_manifest_df, val_manifest_df = \
                data_utils.split_dataframe(snapshot_df,
                                           mode='class_count',
                                           split_class_count=split_count)

        return train_manifest_df, val_manifest_df

    def process_lab(self):
        if self.config['lab'].get('manifest'):
            data_directory = self.config['lab']['manifest']['data_directory']
            train_lab_manifest_path = self.config['lab']['manifest']['train_lab_manifest_path']
            val_lab_manifest_path = self.config['lab']['manifest']['val_lab_manifest_path']

            train_manifest_df = self.process_class_snapshot(train_lab_manifest_path, data_directory)
            val_manifest_df = self.process_class_snapshot(val_lab_manifest_path, data_directory)

        return train_manifest_df, val_manifest_df

    def run(self):
        # Process data given as a directory
        if self.config.get('directory'):
            train_manifest_df, val_manifest_df = self.process_directories()

        # Process data given as a modelmap
        if self.config.get('modelmap'):
            train_manifest_df, val_manifest_df = self.process_modelmap()

        # Process data given as a Lab object
        if self.config.get('lab'):
            train_manifest_df, val_manifest_df = self.process_lab()

        # Store manifests
        train_class_manifest_path = os.path.join(self.output_dir, 'train_class_manifest.json')
        train_manifest_df.to_json(train_class_manifest_path, orient='records', indent=2)

        val_class_manifest_path = os.path.join(self.output_dir, 'val_class_manifest.json')
        val_manifest_df.to_json(val_class_manifest_path, orient='records', indent=2)

        return train_class_manifest_path, val_class_manifest_path


class EvalData(Data):
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = os.path.join(output_dir, 'eval')
        utils.mkdir(self.output_dir)

    def process_directories(self):
        # Read `test_dir` convert to dataframe
        manifest_df = self.process_directory(self.config['directory']['test_dir'])
        return manifest_df

    def process_modelmap(self):
        # Read from config file
        data_directory = self.config['modelmap']['data_directory']
        conditions_manifest = self.config['modelmap']['conditions_manifest_path']
        snapshot_manifest_path = self.config['modelmap']['dataset_manifest_path']
        manifest_df = self.process_modelmap(snapshot_manifest_path, data_directory, conditions_manifest)
        return manifest_df

    def process_lab(self):
        if self.config['lab'].get('manifest'):
            data_directory = self.config['lab']['manifest']['data_directory']
            snapshot_manifest_path = self.config['lab']['manifest']['test_lab_manifest_path']

            manifest_df = self.process_class_snapshot(snapshot_manifest_path, data_directory)

        return manifest_df

    def run(self):
        # Process data given as a directory
        if self.config.get('directory'):
            data_manifest_df = self.process_directories()

        # Process data given as a modelmap
        if self.config.get('modelmap'):
            data_manifest_df = self.process_modelmap()

        # Process data given as a Lab object
        if self.config.get('lab'):
            data_manifest_df = self.process_lab()

        # Store manifests
        test_class_manifest_path = os.path.join(self.output_dir, 'test_class_manifest.json')
        data_manifest_df.to_json(test_class_manifest_path, orient='records', indent=2)

        return test_class_manifest_path
