import warnings
import numpy as np
import pandas as pd
import studio.evaluation.keras.utils as utils

from studio.evaluation.keras.evaluators import CNNEvaluator


class CNNTagEvaluator(CNNEvaluator):
    def __init__(self, **options):
        super().__init__(**options)

    def predict(self):
        raise NotImplementedError("The `predict` function of the {} class is not implemented. Use its `evaluate` function instead".format(self.__class__.__name__))

    @staticmethod
    def get_tag_value(tag_list, tag):
        """ Returns the value of a `tag` present in a list of tags `tag_list`
        """
        tag_value = [t for t in tag_list if t.startswith(tag)]
        if len(tag_value) > 0:
            return tag_value[0].split(":")[1]
        return None

    def evaluate(self, dataframe_path, data_dir='', tag="case-id", top_k=1, filter_indices=None, confusion_matrix=False, custom_crop=True,
                 interpolation='nearest', data_augmentation=None, save_confusion_matrix_path=None, show_confusion_matrix_text=True, validate_filenames=False):
        """Evaluate the performance of the model(s) by tag value.
        Args:
            dataframe_path: Dataframe path
            data_dir: Data directory to load the images from
            tag: Tag to group the evaluation results by
            top_k: An integer specifying the top-k predictions to consider, e.g., top_k = 5 is top-5 preds
            filter_indices: If given take only the predictions corresponding to that indices to compute metrics
            confusion_matrix: If True, show the confusion matrix
            custom_crop: If True, data generator will crop images.
            data_augmentation: Apply the specified augmentation to the image,
                where `data_augmentation` is a dictionary consisting of 3 elements:
                - 'scale_sizes': 'default' (4 scales similar to Going Deeper with Convolutions work) or a list of sizes.
                    Each scaled image then will be cropped into three square parts.
                - 'transforms': list of transforms to apply to these crops in addition to not applying any transform
                ('horizontal_flip', 'vertical_flip', 'rotate_90', 'rotate_180', 'rotate_270' are currently supported).
                - 'crop_original': 'center_crop' mode allows to center crop the original image prior to performing the
                transforms, scalings, and croppings.
            save_confusion_matrix_path: File name and path where to save the confusion matrix
            show_confusion_matrix_text: If False, will hide the text in the confusion matrix
            interpolation: String indicating the interpolation parameter for the data generator.
            validate_filenames: If True, images with invalid filename extensions will be ignored.
        Returns: A dictionary of the computed metrics between the predicted probabilities and ground truth labels.
        """

        self.top_k = top_k
        self.data_dir = data_dir
        self.dataframe = pd.read_json(dataframe_path)

        # Add a tag column in the dataframe to compute evaluation results per tag
        self.dataframe[tag] = self.dataframe.apply(lambda row: self.get_tag_value(row['tags'], tag), axis=1)

        if self.concepts is None:
            self.concepts = [{'id': 'class_' + str(i), 'label': 'class_' + str(i)}
                             for i in range(len(self.dataframe['class_probabilities'][0]))]

        self.concept_labels = utils.get_concept_items(self.concepts, key='label')

        # Create Keras image generator and obtain probabilities
        probabilities, self.labels_categorical = self._compute_probabilities_generator(
            data_dir=data_dir, dataframe=self.dataframe, custom_crop=custom_crop, interpolation=interpolation,
            data_augmentation=data_augmentation, validate_filenames=validate_filenames)

        # Collapse probabilities, obtain 1D label array
        self.probabilities = utils.combine_probabilities(probabilities, self.combination_mode)

        if hasattr(self, 'concept_dictionary'):
            if utils.compare_group_test_concepts(self.concept_labels, self.concept_dictionary) \
                    and utils.check_concept_unique(self.concept_dictionary):

                self.probabilities = self._compute_inference_probabilities(self.probabilities)
            else:
                # Should never be here, but added in case the `utils` function one day fails.
                raise ValueError("Error: Invalid `concept_dictionary`.")

        self.labels = self.labels_categorical.argmax(axis=1)
        self.tags = self.dataframe[tag]

        # Separate the index of cases without tags (i.e. tag value is None) from those with tags
        none_tags_indices = self.tags[self.tags.isna()].index.values
        valued_tags_indices = self.tags[~self.tags.isna()].index.values

        # Group the cases with `None` tags
        none_tags_probabilities = self.probabilities[none_tags_indices, :]
        none_tags_labels = self.labels[none_tags_indices]

        # Group the cases with valued tags by tag. Every image in a group keeps its original row index in `self.dataframe`
        # to map it to its corresponding probability and label in `self.probabilities` and `self.labels`
        valued_tags_dataframe = self.dataframe.loc[valued_tags_indices]
        valued_tag_grouped = valued_tags_dataframe.groupby(tag)

        # Group by the cases with valued tags, average their probabilities and save the cases
        # having images with different labels in a dedicated dataframe
        # 1. Define intermediate variables
        # the list containing the indices of the cases having having images with different labels
        different_label_cases_indices = []
        # the list of mean probabilities of cases
        grouped_probabilities = []
        # the list of the labels of cases for those having images with the same label
        grouped_labels = []

        # 2. Loop over every tag group
        for tag_value, tag_group in valued_tag_grouped:
            # get the indices of the group
            group_indices = tag_group.index.values
            # get the probabilities and the labels of the group
            group_probabilities = self.probabilities[group_indices, :]
            group_labels = self.labels[group_indices]
            # average the probabilities of the group
            mean_group_probabilities = np.mean(group_probabilities, axis=0)
            # save the group indices if the group labels are not equal
            if len(set(group_labels)) > 1:
                different_label_cases_indices.append(list(group_indices))
                warnings.warn("The images with the tag value `{}` have different labels".format(tag_value))
                # go to the next group
                continue

            # append the representative probability and label of the group
            grouped_probabilities.append(mean_group_probabilities)
            grouped_labels.append(group_labels[0])

        # 3. Put the cases with multiple labels in a dataframe
        different_label_cases_indices = sum(different_label_cases_indices, [])
        self.different_label_cases_df = self.dataframe.loc[different_label_cases_indices]

        # cast the grouped probabilities and labels to numpy arrays
        grouped_probabilities = np.array(grouped_probabilities)
        grouped_labels = np.array(grouped_labels)
        # add the `None` probabilities and labels to the valued ones
        self.probabilities = np.concatenate((grouped_probabilities, none_tags_probabilities), axis=0)
        self.labels = np.concatenate((grouped_labels, none_tags_labels), axis=0)

        self.results = self.get_metrics(probabilities=self.probabilities, labels=self.labels,
                                        concepts=self.concepts, top_k=self.top_k,
                                        filter_indices=filter_indices,
                                        confusion_matrix=confusion_matrix,
                                        save_confusion_matrix_path=save_confusion_matrix_path,
                                        show_confusion_matrix_text=show_confusion_matrix_text)
        return self.results
