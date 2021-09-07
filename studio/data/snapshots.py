import os
import json
import operator
import numpy as np
import pandas as pd
import studio.data.utils as utils


from collections import Counter
from studio.utils.utils import mkdir
from studio.data.review_algorithms import compute_average_review


class Snapshot(object):
    def __init__(self,
                 snapshot_manifest,
                 root_directory=''):
        """
        Process Lab snapshot manifests.

        Args:
            snapshot_manifest: The JSON file(s) that contains the Lab snapshot manifest(s).
                The input can be,
                - a pandas dataframe
                - a string indicate the JSON file to load
                    `snapshot_manifest = 'file1.json'`
                - a list of JSON filenames to load
                    `snapshot_manifest = ['file1.json', 'file2.json']`
            root_directory: Folder directory containing all the images (Optional)
                            Useful if the manifest only contains relative paths.
        """

        self.root_directory = root_directory

        # Convert to a list if is a string.
        if isinstance(snapshot_manifest, str):
            snapshot_manifest = [snapshot_manifest]

        if isinstance(snapshot_manifest, pd.DataFrame):
            # Use the dataframe directly.
            self.manifest_df = snapshot_manifest
        elif isinstance(snapshot_manifest, list):
            # Load and concatenate all the JSON files to form a single file.
            snapshot_manifests = []
            for manifest_path in snapshot_manifest:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    snapshot_manifests.extend(manifest)

            self.manifest_df = pd.json_normalize(snapshot_manifests)
        else:
            raise ValueError("Error: `snapshot_manifest` is an unexpected data type.")

        # If a dictionary is present, set the class dictionary, else set to None.
        self.class_dictionary_df = self._to_class_dictionary_df(self.manifest_df)
        self._add_labels_source_and_type()

    def _add_labels_source_and_type(self):
        self.manifest_df['labels_source'] = None
        self.manifest_df['image_type'] = None
        self.rows_tie_image_type = []
        self.rows_missing_image_type = []
        self.rows_missing_AIP_code = []
        self.rows_multiple_sources = []
        self.rows_multiple_AIP_code = []
        for index, row in self.manifest_df.iterrows():
            if 'reviews' in self.manifest_df.columns and isinstance(row['reviews'], list):
                self.manifest_df.at[index, 'labels_source'] = 'review_pipeline'
                # Check for image type
                image_type_array = []
                for i, reviews in enumerate(row['reviews']):
                    if 'image_type' in reviews:
                        image_type_array.append(reviews['image_type'])
                image_type_array_counts = Counter(image_type_array)
                if len(image_type_array_counts) > 0:
                    img_type = image_type_array_counts.most_common()[0][0]
                    if len(image_type_array_counts) > 1:
                        top_2_counts = image_type_array_counts.most_common(2)
                        # If there is a tie, we don't assign image type
                        if top_2_counts[0][1] == top_2_counts[1][1]:
                            self.rows_tie_image_type.append(index)
                            img_type = None

                    self.manifest_df.at[index, 'image_type'] = img_type

                if 'tags' in self.manifest_df.columns:
                    tag = utils.search_tags(row['tags'], tags='AIP')
                    if len(tag) > 0:
                        self.manifest_df.at[index, 'labels_source'] = 'multiple_sources'
                        self.rows_multiple_sources.append(index)
            else:
                if 'tags' in self.manifest_df.columns:
                    tag = utils.search_tags(row['tags'], tags='AIP')
                    if len(tag) > 0:
                        self.manifest_df.at[index, 'labels_source'] = 'AIP'
                        if len(tag) > 1:
                            self.rows_multiple_AIP_code.append(index)
                    else:
                        self.rows_missing_AIP_code.append(index)

                    tag = utils.search_tags(row['tags'], tags='image-type')
                    if len(tag) > 0:
                        if len(tag) > 1:
                            self.rows_tie_image_type.append(index)
                        else:
                            self.manifest_df.at[index, 'image_type'] = tag[0].replace('image-type:', '')
                    else:
                        self.rows_missing_image_type.append(index)
                else:
                    self.rows_missing_AIP_code.append(index)

        # self.manifest_df = self.manifest_df.drop(drop_indices).reset_index().drop(columns='index')

        if len(self.rows_tie_image_type) > 0:
            print('There have been %i rows with a tie in the image type. Access the indices in '
                  'snapshot.rows_tie_image_type' % (len(self.rows_tie_image_type)))

        if len(self.rows_missing_image_type) > 0:
            print('There have been %i rows missing image type. Access the indices in '
                  'snapshot.rows_missing_image_type' % (len(self.rows_missing_image_type)))

        if len(self.rows_missing_AIP_code) > 0:
            print('There have been %i rows missing AIP codes. Access the indices in '
                  'snapshot.rows_missing_AIP_code' % (len(self.rows_missing_AIP_code)))

        if len(self.rows_multiple_sources) > 0:
            print('There have been %i rows containing a AIP code and a review. Access the indices in '
                  'snapshot.rows_multiple_sources' % (len(self.rows_multiple_sources)))

        if len(self.rows_multiple_AIP_code) > 0:
            print('There have been %i rows containing rows_multiple AIP codes. Access the indices in '
                  'snapshot.rows_multiple_AIP_code' % (len(self.rows_multiple_AIP_code)))

    def get_reviews(self, reviewer):
        out_reviews = []
        for index, row in self.manifest_df.iterrows():
            if 'reviews' in self.manifest_df.columns and isinstance(row['reviews'], list):
                for i, reviews in enumerate(row['reviews']):
                    if reviews['reviewer_email'] == reviewer:
                        out_reviews.append(reviews['diagnoses'])

        print('Reviewer %s had %i reviews ' % (reviewer, len(out_reviews)))

        return out_reviews

    def exclude_reviewers(self, reviewers):
        if isinstance(reviewers, str):
            reviewers = [reviewers]

        reviewers_exclusions = {reviewer: 0 for reviewer in reviewers}
        for index, row in self.manifest_df.iterrows():
            if 'reviews' in self.manifest_df.columns and isinstance(row['reviews'], list):
                new_reviews = []
                for i, reviews in enumerate(row['reviews']):
                    if reviews['reviewer_email'] not in reviewers_exclusions:
                        new_reviews.append({'reviewer_email': reviews['reviewer_email'],
                                            'diagnoses': reviews['diagnoses']})
                    else:
                        reviewers_exclusions[reviews['reviewer_email']] += 1
                self.manifest_df.at[index, 'reviews'] = new_reviews

        for reviewer, n_exclusions in reviewers_exclusions.items():
            print('Reviewer %s excluded from %i reviews ' % (reviewer, n_exclusions))

    @staticmethod
    def _to_class_dictionary_df(manifest_df):
        """
        Convert `manifest_df` to a class dictionary dataframe format.
        Args:
            manifest_df: A dataframe containing the Lab snapshot manifest with class 'dictionary' information.
        Returns:
            If the dictionary is not present within `manifest_df`, returns None.
            Else returns,
            class_dictionary_df: A dataframe containing the Lab snapshot manifest information organized by classes.
                                Each dictionary class contains 'class_index', 'class_name' and image 'count'.
        """
        dictionary_keys = ['dictionary.class_index', 'dictionary.class_name']

        if not any(dictionary_key in manifest_df.keys() for dictionary_key in dictionary_keys):
            # raise ValueError('`manifest_df` does not contain any class dictionary values.')
            return None

        class_indices = []
        class_names = []
        counts = []

        # Get existing class indexes. This may not be comprehensive e.g., index of 3 may exist, but 2 may not.
        class_index_range = manifest_df['dictionary.class_index'].unique()

        for class_index in class_index_range:
            class_df = manifest_df.loc[manifest_df['dictionary.class_index'] == class_index]
            (class_name,) = set(class_df['dictionary.class_name'].values)
            class_count = len(class_df)

            class_indices.append(class_index)
            class_names.append(class_name)
            counts.append(class_count)

        data = {
            'class_index': class_indices,
            'class_name': class_names,
            'count': counts
        }

        class_dictionary_df = pd.DataFrame(data)
        class_dictionary_df = class_dictionary_df[['class_index', 'class_name', 'count']]

        return class_dictionary_df

    def compute_node_frequency(self, uncertainty_threshold=1.0, uncertainty_mode='keep'):
        """

        Args:
            uncertainty_threshold: We will discard samples with an average uncertainty higher or equal than
            `uncertainty_threshold` parameter

        Returns:  Returns: A dataframe containing `node_id` and `frequency` counts for all the samples that appear in
        the manifest

        """
        # Initialize dictionary
        node_frequency = {}
        if 'average_reviews' not in self.manifest_df.columns:
            self.compute_average_reviews(uncertainty_mode=uncertainty_mode)

        for reviews in self.manifest_df['average_reviews']:
            # We consider the review if uncertainty is below the uncertainty_threshold
            if 'uncertainty' in reviews and reviews['uncertainty'] >= uncertainty_threshold:
                pass
            else:
                for node, frequency in reviews.items():
                    node_frequency[node] = frequency if node not in node_frequency else node_frequency[node] + frequency

        sorted_counter = sorted(node_frequency.items(), key=operator.itemgetter(1), reverse=True)
        df = pd.DataFrame({'node_id': list(dict(sorted_counter).keys()),
                           'frequency': list(dict(sorted_counter).values())})
        return df

    def compute_average_reviews(self,
                                bad_quality_tag='uncertainty',
                                uncertainty_mode='keep',
                                tag_key='AIP'):
        """

        Args:
            bad_quality_tag: Tag to apply to images with a bad quality review
            uncertainty_mode: Accepts `drop` (discard uncertainty), `keep`(keep uncertainty value in the
            review dictionary) and `distribute` (distribute the uncertainty between other diagnosis) as values. Used
            when computing average reviews.
            tag_key: Tag containing the sample label (used when there are no reviews).

        Returns:

        """

        """
        Returns: The original snapshot manifest dataframe plus a column named `average_reviews`
        containing a dictionary with the average reviews or the original label (if there were no reviews)
        """
        self.manifest_df['average_reviews'] = None
        for index, row in self.manifest_df.iterrows():
            # If the sample has reviews, compute the average review
            reviews = []
            if row['labels_source'] == 'review_pipeline':
                for review in row['reviews']:
                    diagnoses = review['diagnoses']
                    if len(diagnoses) < 1:
                        # Setting BAD QUALITY to uncertainty = 100 until we decide what to do
                        diagnoses = {bad_quality_tag: 100}
                    reviews.append(diagnoses)
                average_reviews = compute_average_review(reviews, uncertainty_mode)
            # If not, search for `tag_key` codes
            else:
                tag = utils.search_tags(row['tags'], tags=tag_key)
                if len(tag) > 0:
                    average_reviews = {tag[0]: 1.0}
                # If sample does not have a review or `tag_key` code we will append an empty dictionary
                else:
                    average_reviews = {}
            self.manifest_df.at[index, 'average_reviews'] = average_reviews

        return self.manifest_df

    def set_root_directory(self, directory):
        self.root_directory = directory

    def resize_and_save(self, output_directory, apply_crop=True, size=(512, 512), keep_aspect_ratio=True,
                        resize_if_smaller=False, column_id='storage_key'):
        """

        Args:
            output_directory: Directory where to save the new images
            apply_crop: If True will apply crop under dataframe `crop` column
            size: Tuple (width, height) to which the image will be resized
            keep_aspect_ratio: If True will keep aspect ratio of original image, resizing to the minimum size
            resize_if_smaller: If True will not resize the image if it is already smaller than `size`
            column_id: dataframe column key containing the image path

        Returns: Manifest dataframe with filename changed to the new paths and updated width and height

        """

        df = self.manifest_df.copy()
        for index, row in self.manifest_df.iterrows():
            image = utils.load_image(os.path.join(self.root_directory, row[column_id]))
            if apply_crop and isinstance(row['crop'], list):
                image = utils.apply_crop(image, row['crop'])
            if size is not None:
                image = utils.resize_image(image, size, keep_aspect_ratio, resize_if_smaller)
            # If image directory contains sub-folders
            save_dir = os.path.join(output_directory, "".join(row[column_id].split('/')[:-1]))
            mkdir(save_dir)
            file_path = os.path.join(output_directory, row[column_id])
            image.save(file_path, format='jpeg')
            df.at[index, 'filename'] = file_path
            df.at[index, 'width'], df.at[index, 'height'] = image.size
        return df

    def process_filename_column(self, column_id='storage_key'):
        """
        Will join self.manifest_df[column_id]] to `root_directory` (if specified) and assign it to the column 'filename'
        Args:
            root_directory: Root folder where images are stored
            column_id: Column id of dataframe containing relative paths for images

        Returns: A manifest dataframe with the filename format ready for training and evaluating

        """
        if self.root_directory is not None:
            filenames = [os.path.join(self.root_directory, filename) for filename in self.manifest_df[column_id]]
            self.manifest_df['filename'] = filenames

        return self.manifest_df

    def append_class_probabilities(self):
        # Check if the manifest has the `dictionary.class_index` key
        if 'dictionary.class_index' in self.manifest_df.columns.tolist():
            # Get the class count
            class_count = len(self.manifest_df['dictionary.class_index'].unique())
            # Append the one hot encoding vector of every class number to the manifest dataframe
            self.manifest_df['class_probabilities'] = self.manifest_df.apply(lambda row: np.eye(class_count)[row['dictionary.class_index']], axis=1)

            return self.manifest_df
        else:
            raise ValueError("Error: the lab manifest does not contain the `dictionary` key.")
