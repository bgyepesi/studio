import json
import numpy as np
import studio.data.utils as utils

from studio.data.review_algorithms import compute_average_review, reject_outliers, outliers_as_uncertainty, \
    uncertainty_is_above_threshold


class ModelMap:
    """
    Helper Class that is used to map AIP codes to model outputs.
    It can also be used for documentation and visualization purposes.
    """

    def __init__(self, conditions_df):
        """Mapping among the conditions, diagnosis, and class indexes.

        Args:
            conditions_df: dataframe storing the model map data
        """
        self.conditions_df = conditions_df
        self._init_attrs()

    def _init_attrs(self):

        self.diagnosis_df = self.conditions_df.groupby(
            ['class_index', 'diagnosis_id', 'diagnosis_name', 'malignancy'],  # Columns to keep.
            as_index=False,  # Needed for indexing later.
        ).agg({
            'condition_id': list,  # Append all condition IDs.
            'condition_name': list,  # Append all condition names.
            'n_samples': 'sum',  # Sum the number of samples.
        }
        )
        if len(self.diagnosis_df) != len(self.conditions_df.class_index.unique()):
            # If the group-by values are not all the same
            # (e.g., a diagnosis node maps to a benign and a malignant condition)
            # additional rows will be created and will not have a 1-to-1 relationship with the class indexes.
            raise ValueError("Error: The number of diagnosis nodes differ from the number of unique class indexes.")

        # List of diagnosis IDs
        self.diagnosis_ids = self.diagnosis_df['diagnosis_id'].tolist()
        # List of condition IDs
        self.condition_ids = self.conditions_df['condition_id'].tolist()
        # Dictionary mapping each condition to its diagnosis node
        self.conditions2diagnosis_map = {}
        # Dictionary mapping each condition to its label
        self.conditions_id2name_map = {}
        # Dictionary mapping each diagnosis to its class index
        self.diagnosis2class_index_map = {}
        # Dictionary mapping each diagnosis to its label
        self.diagnosis_id2name_map = {}

        for _, row in self.conditions_df.iterrows():
            self.conditions2diagnosis_map[row['condition_id']] = row['diagnosis_id']
            self.conditions_id2name_map[row['condition_id']] = row['condition_name']

        for _, row in self.diagnosis_df.iterrows():
            self.diagnosis2class_index_map[row['diagnosis_id']] = row['class_index']
            self.diagnosis_id2name_map[row['diagnosis_id']] = row['diagnosis_name']

    def map_reviews(self, manifest_df, mode=None, mapping_function=None):
        """
        Args:
            mode: All modes will merge children into diagnosis nodes following the model map given
                1. `outliers_as_uncertainty`: will map outliers choices to uncertainty
                2. `reject_outliers` will reject reviews individually with at least one outlier inside.
                    If a review is empty after that, sample will be discarded
            mapping_function: f(diagnoses: Dict, model_map: ModelMap) --> diagnoses: Dict

        Returns: A mapped manifest df with a `mapped_reviews` column which corresponds to a dictionary of `reviews`
        given by reviewers mapped to the model map diagnosis node classes.
        If we only had a AIP code, it will be mapped to its diagnosis node.

        """

        if mode is not None:
            if mode not in ('outliers_as_uncertainty', 'reject_outliers'):
                raise ValueError('`mode` must be either `outliers_as_uncertainty` or `reject_outliers`')
            else:
                if mode == 'reject_outliers':
                    mapping_function = reject_outliers
                elif mode == 'outliers_as_uncertainty':
                    mapping_function = outliers_as_uncertainty
        else:
            if mapping_function is None:
                raise ValueError('`mode` and `mapping_function` are both None. Specify a value for one of them')

        mapped_manifest_df = manifest_df.copy(deep=True)
        mapped_manifest_df['mapped_reviews'] = None
        drop_indices = []
        for index, row in mapped_manifest_df.iterrows():
            mapped_reviews = []
            # If has multiple sources we default to reviews
            if row['labels_source'] in ('review_pipeline', 'multiple_sources'):
                for review in row['reviews']:
                    mapped_review = {'reviewer_email': review['reviewer_email'], 'diagnoses': {}}
                    diagnoses = review['diagnoses']
                    if len(diagnoses) < 1:
                        # Setting Bad quality to uncertainty = 100 until we decide what to do
                        diagnoses = {'uncertainty': 100}
                    mapped_review['diagnoses'] = mapping_function(diagnoses, self)
                    if len(mapped_review['diagnoses']) > 0:
                        mapped_reviews.append(mapped_review)
            else:
                # If not, search for `tag_key` codes
                tag = utils.search_tags(row['tags'], tags='AIP')
                if len(tag) > 0:
                    review = {tag[0]: 100}
                    diagnosis = mapping_function(review, self)
                    # Means that the AIP tag was not assigned to anything so it should be discarded
                    if 'uncertainty' in diagnosis or len(diagnosis) < 1:
                        mapped_reviews = []
                    else:
                        mapped_reviews = [{'diagnoses': diagnosis}]
                # If sample does not have a review or `tag_key` code we will append an empty dictionary
                else:
                    mapped_reviews = []

            if len(mapped_reviews) > 0:
                mapped_manifest_df.at[index, 'mapped_reviews'] = mapped_reviews
            else:
                drop_indices.append(index)

        print('%i samples were dropped due to a lack of reviews or AIP code' % len(drop_indices))

        mapped_manifest_df = mapped_manifest_df.drop(drop_indices).reset_index().drop(columns='index')

        return mapped_manifest_df

    def map_manifest(self,
                     manifest_df,
                     mapping_mode=None,
                     mapping_function=None,
                     min_reviews=1,
                     uncertainty_as_class=False,
                     uncertainty_mode='keep',
                     uncertainty_threshold=1.0
                     ):
        """

        Args:
            snapshot_df: Snapshot dataframe.
            mapping_mode: All modes will merge children into diagnosis nodes following the model map
                1. `outliers_as_uncertainty`: will map outliers choices to uncertainty
                2. `reject_outliers` will reject reviews individually with at least one outlier inside.
                    If a review is empty after that, sample will be discarded
            mapping_function: Custom mapping function f(diagnoses: Dict, model_map: ModelMap) --> diagnoses: Dict
            min_reviews: A sample will be only kept if it has a `min_reviews` number. Only applies to samples coming
            from the review pipeline
            uncertainty_as_class: If True, will add an extra uncertainty class to the model map and assign labels to it
            uncertainty_threshold: We will discard samples with an average uncertainty higher or equal than
            `uncertainty_threshold` parameter
            uncertainty_mode: Accepts `drop` (discard uncertainty), `keep`(keep uncertainty value in the
            review dictionary) and `distribute` (distribute the uncertainty between other diagnosis) as values. Used
            when computing average reviews.

        Returns:
            This function will process a dataset manifest and return a `mapped_manifest` with extra columns:

            1. `mapped_reviews` which corresponds to a dictionary of `reviews` given by reviewers mapped to the model
            map diagnosis node classes. If we only had a aip code, it will be mapped to its diagnosis node.
            2. `class_probabilities` which corresponds to a list of probabilities sorted by the model_map `diagnosis_df`
            class indices.

        """
        manifest_df['class_probabilities'] = None
        drop_indices_uncertainty, drop_indices_min_reviews, class_probabilities = [], [], []

        if uncertainty_as_class:
            if uncertainty_mode != 'keep':
                raise ValueError('If `uncertainty_as_class` is True, `uncertainty_mode` must be `keep`')
            self.conditions_df = self.conditions_df.append({
                'class_index': max(self.conditions_df.class_index) + 1,
                'condition_id': 'uncertainty',
                'condition_name': 'uncertainty',
                'diagnosis_id': 'uncertainty',
                'diagnosis_name': 'uncertainty',
                'n_samples': 0,
                'malignancy': 'benign'
            },
                ignore_index=True
            )
            self._init_attrs()

        mapped_manifest_df = self.map_reviews(manifest_df, mode=mapping_mode, mapping_function=mapping_function)

        mapped_manifest_df['average_diagnosis_reviews'] = None

        for index, row in mapped_manifest_df.iterrows():
            review = []
            for reviews in row['mapped_reviews']:
                review.append(reviews['diagnoses'])

            # Perform check before performing average review, as some options can skip avg uncertainty (e.g. distribute)
            if uncertainty_is_above_threshold(review, uncertainty_threshold):
                drop_indices_uncertainty.append(index)
            # If we have review and AIP codes we will default to process reviews
            # If the review has only been reviewed less times than min_reviews or it is empty, discard
            elif (row['labels_source'] in ('review_pipeline', 'multiple_sources') and len(review) < min_reviews) \
                    or len(review) < 1:
                drop_indices_min_reviews.append(index)
            else:
                average_review = compute_average_review(review, uncertainty_mode=uncertainty_mode)
                mapped_manifest_df.at[index, 'average_diagnosis_reviews'] = average_review
                class_probabilities.append(self.diagnosis2probability(average_review,
                                                                      uncertainty_as_class=uncertainty_as_class))

        drop_indices = drop_indices_min_reviews + drop_indices_uncertainty
        mapped_manifest_df = mapped_manifest_df.drop(drop_indices).reset_index()
        mapped_manifest_df = mapped_manifest_df.assign(class_probabilities=class_probabilities)

        print('%i samples were discarded because the number of reviews was below %i and %i because of high uncertainty '
              % (len(drop_indices_min_reviews), min_reviews, len(drop_indices_uncertainty)))

        return mapped_manifest_df

    def diagnosis2probability(self, diagnosis_probabilities, uncertainty_as_class=False, dictionary=False):
        """
        Map a dictionary of diagnosis with probabilities to a class probability array.
        Args:
            diagnosis_probabilities: A dictionary {diagnosis: probability}
            uncertainty_as_class: If True will create a separate class for uncertainty, else it will distribute
            uncertainty evenly to all the diagnosis nodes that had probability 0.
            dictionary: If True, it will return a dictionary, else an ordered list of probabilities
            (by class indices)

        Returns:
            A class probability array
        """

        # Initialize variables
        other_diagnosis_uncertainty_probability = 0
        class_probabilities = {diagnosis_id: 0 for diagnosis_id in self.diagnosis_ids}
        diagnosis_count = 0

        for diagnosis_id, prob in diagnosis_probabilities.items():
            if not uncertainty_as_class and diagnosis_id == "uncertainty":
                other_diagnosis_uncertainty_probability = prob
            else:
                class_probabilities[diagnosis_id] += prob
                diagnosis_count += 1

        # If there was uncertainty
        if other_diagnosis_uncertainty_probability > 0:
            # If there are more diagnosis nodes not included in the reviews distribute among them
            if len(class_probabilities) - diagnosis_count > 0:
                other_diagnosis_uncertainty_probability /= (len(class_probabilities) - diagnosis_count)
                for diagnosis, probability in class_probabilities.items():
                    if probability == 0:
                        class_probabilities[diagnosis] = other_diagnosis_uncertainty_probability
            # If all the diagnosis nodes are included in the review distribute the uncertainty among them
            else:
                other_diagnosis_uncertainty_probability /= len(class_probabilities)
                for diagnosis, probability in class_probabilities.items():
                    class_probabilities[diagnosis] += other_diagnosis_uncertainty_probability
        if dictionary:
            return class_probabilities
        else:
            return np.array([class_probabilities[diagnosis] for diagnosis in self.diagnosis_ids])

    def save(self, filename, mode='conditions', format='csv', columns=None):
        supported_modes = ('diagnosis', 'conditions')
        supported_format = ('csv', 'json')

        # Select data
        if mode == 'diagnosis':
            df = self.diagnosis_df
        elif mode == 'conditions':
            df = self.conditions_df
        else:
            raise ValueError('Mode %s inputed not valid, the valid ones are %s' % (mode, supported_modes))

        # Now take a subset of the specified columns.
        columns = columns or df.columns
        df = df[columns]

        # Save in selected format
        if format == 'csv':
            filename += '.csv'
            df.to_csv(filename, index=False)
        elif format == 'json':
            df_json = []
            filename += '.json'
            for index, row in df.iterrows():
                diagnosis_attributes = {}
                for column in columns:
                    diagnosis_attributes[column] = row[column]

                df_json.append(diagnosis_attributes)

            with open(filename, 'w') as outfile:
                json.dump(df_json, outfile, sort_keys=True, indent=4)
            return df_json
        else:
            raise ValueError('Format %s inputed not valid, the valid ones are %s' % (format, supported_format))
