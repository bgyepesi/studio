import os
import copy
import json
import numpy as np
import studio.evaluation.keras.utils as utils
import studio.evaluation.keras.metrics as metrics

from studio.evaluation.keras.data import QAData
from visual_qa.visual_qa import VisualQA
from studio.evaluation.keras.evaluators import Evaluator


class VisualQAEvaluator(Evaluator):
    """
    This class is responsible of the evaluation of Visual-Q&A system over a test set.
    """
    OPTIONS = {
        'id': {'type': str, 'default': 'visual_qa_evaluator'},
        'report_dir': {'type': str},
        'by_definition_csv': {'type': str},
        'visual_dictionary': {'type': str},
        'qa_data_json': {'type': str},
        'valid_evidence': {'type': str},
        'mode': {'type': str, 'default': 'combined'},
        'image_paths': {'type': list, 'default': None}
    }

    def __init__(self, **options):
        self.results = {}
        self.condition_rank = []

        for key, option in self.OPTIONS.items():
            if key not in options and 'default' not in option:
                raise ValueError('missing required option: %s' % (key,))
            value = options.get(key, copy.copy(option.get('default')))
            setattr(self, key, value)

        self.qa_data = QAData(self.qa_data_json)

        self.supported_modes = ["inclusion", "exclusion", "combined"]

        self.visual_qa = VisualQA(self.valid_evidence, self.by_definition_csv, self.visual_dictionary)

        self.filtered_qa_data = self.qa_data.filter_diagnosis_ids(list(self.visual_qa.general.diagnoses))

        self.valid_evidence_data = utils.read_json(self.valid_evidence)

        self.concepts = [{'id': row['condition_id'],
                          'label': row['diagnosis_name'],
                          'diagnosis_id': row['diagnosis_id']}
                         for i, row in enumerate(self.visual_qa.general.visual_dictionary)]

        # Obtain labels to show on the metrics results
        self.concept_labels = utils.get_concept_items(self.concepts, key='label')

    def evaluate(self,
                 prior_probabilities,
                 strict_mode,
                 top_k=5,
                 mode='combined',
                 report=True):
        """
        This function evaluates the Q&A system.
        Args:
            prior_probabilities: An array of prior_probabilities based in the order of the test set.
            strict_mode: If true, checks that all evidence value can be possible in the matrix. If false,
                        checks if the any evidence for a particular is true.
            top_k: The top-k predictions to consider. E.g. top_k = 5 is top-5 preds
            mode: VisualQA mode, supported modes are: `inclusion`, 'exclusion` and `combined`
            differential: If true, compute differential quality score.
            ranked_differential: If true, compute differential quality score on ranked differentials.
            report: A boolean which determines whether the evaluation report should be stored.

        Returns: Performance metrics of the system. Saves array of predicted probabilties and labels.

        """
        labels = []

        if mode not in self.supported_modes:
            raise ValueError("%s mode not supported" % mode)

        self.mode = mode
        self.prior_probabilities = prior_probabilities
        self.probabilities = []
        diagnosis_ids = list(self.visual_qa.general.diagnoses)

        for case_index in range(len(self.filtered_qa_data)):
            self.filtered_qa_data[case_index]['evidence'].update({
                'visual_probabilities': prior_probabilities[case_index].tolist()})

            labels.append(diagnosis_ids.index(self.filtered_qa_data[case_index]['diagnosis_id']))

            try:
                if self.mode == "inclusion":
                    filtered_probs = self.visual_qa.filter_general(self.filtered_qa_data[case_index]['evidence'],
                                                                   inclusion=True,
                                                                   exclusion=False)
                elif self.mode == "exclusion":
                    filtered_probs = self.visual_qa.filter_general(self.filtered_qa_data[case_index]['evidence'],
                                                                   inclusion=False,
                                                                   exclusion=True,
                                                                   strict_mode=strict_mode)
                else:
                    filtered_probs = self.visual_qa.filter_general(self.filtered_qa_data[case_index]['evidence'],
                                                                   inclusion=True,
                                                                   exclusion=True,
                                                                   strict_mode=strict_mode)
            except ValueError:
                filtered_probs = np.array(self.prior_probabilities[case_index])

            self.probabilities.append(filtered_probs)

        self.probabilities = np.array(self.probabilities)
        self.labels = np.array(labels)
        self.results = metrics.metrics_top_k(self.probabilities,
                                             self.labels,
                                             self.concepts, top_k)

        if report:
            self.report_results(self.labels, self.probabilities)

        return self.results

    def predict(self, prior_probabilities, strict_mode, mode='combined'):
        self.mode = mode
        self.prior_probabilities = prior_probabilities
        self.probabilities = []
        differential_diagnosis = []

        for case_index in range(len(self.filtered_qa_data)):
            differential_diagnosis.append(self.filtered_qa_data[case_index]['differential'])
            self.filtered_qa_data[case_index]['evidence'].update({
                'visual_probabilities': prior_probabilities[case_index].tolist()})

            if self.mode not in self.supported_modes:
                raise Exception("mode not supported")
            else:
                try:
                    if self.mode == "inclusion":
                        filtered_probs = self.visual_qa.filter_general(self.filtered_qa_data[case_index]['evidence'],
                                                                       inclusion=True,
                                                                       exclusion=False, strict_mode=strict_mode)
                    elif self.mode == "exclusion":
                        filtered_probs = self.visual_qa.filter_general(self.filtered_qa_data[case_index]['evidence'],
                                                                       inclusion=False,
                                                                       exclusion=True, strict_mode=strict_mode)
                    else:
                        filtered_probs = self.visual_qa.filter_general(self.filtered_qa_data[case_index]['evidence'],
                                                                       inclusion=True,
                                                                       exclusion=True, strict_mode=strict_mode)
                except ValueError:
                    filtered_probs = np.array(self.prior_probabilities[case_index])

                self.probabilities.append(filtered_probs)

        return self.probabilities

    def report_results(self, labels, probabilities, errors=True, improvements=True):
        '''
        This function helps in storing the metrics, errors and improvements of the Q&A system.
        Args:
            labels: A list containing the ground truth condition indices.
            probabilities: The list of probabilities which are the output of the visual-qa system.
            errors: A boolean value which determines if the data inconsistencies should be flagged.
            improvements: If true, the Q&A system improvements vs visual probabilities are stored.
        '''

        by_definition_errors, test_matrix_diff_zeros, test_matrix_diff_twos, by_definition_improvements, rank_change = \
            self.find_errors_improvements(labels, probabilities, errors, improvements)

        visual_non_zeros = np.count_nonzero(self.prior_probabilities)
        visual_qa_non_zeroes = np.count_nonzero(self.probabilities)

        avg_result = self.results['average']
        del avg_result['confusion_matrix']
        errors_percentage = (len(by_definition_errors) / len(self.filtered_qa_data)) * 100
        errors_percentage_str = ["Error percentage:" + str(errors_percentage) + "%"]
        avg_rank_change = np.mean(rank_change)
        # Average change in position of the true label before and after the QA system.
        avg_result['average_rank_change'] = avg_rank_change
        # Total number of probabilities that we zeros based on the CNNs output.
        avg_result['Visual_Non_Zero_Probs'] = visual_non_zeros
        # Total number of probabilities that we zeros based on the VisualQA's output.
        avg_result['Visual_QA_Non_Zero_Probs'] = visual_qa_non_zeroes
        # Change in number of zero probabilities before and after the QA system.
        avg_result['Absolute_Zero_change'] = visual_qa_non_zeroes - visual_non_zeros
        # Percentage of change in number of zero probabilities before and after the QA system.
        avg_result['Zero_change_percentage'] = (avg_result['Absolute_Zero_change'] / visual_non_zeros) * 100

        if not os.path.exists(self.report_dir):
            os.mkdir(self.report_dir)

        utils.store_data_csv(self.report_dir, 'vignette_errors.csv', by_definition_errors,
                             extra_string=errors_percentage_str)

        utils.store_data_csv(self.report_dir, 'by_definition_improvements.csv', by_definition_improvements)

        utils.store_data_csv(self.report_dir, 'matrix_errors_zeros.csv', test_matrix_diff_zeros)

        utils.store_data_csv(self.report_dir, 'matrix_errors_twos.csv', test_matrix_diff_twos)

        with open(os.path.join(self.report_dir, 'results.json'), 'w') as fp:
            json.dump(avg_result, fp, indent=4)

    def report_differential(self, avg_precision_differential, quality_score, top_k=5):
        """
        Stores the results of differential evaluation in a text file.
        Args:
            avg_precision_differential : Average precision score for the differential diagnosis.
            quality_score: Score computed using Normalized Discounted Cumulative Gain.
            top_k: The number of differentials considered.
        """
        if not os.path.exists(self.report_dir):
            os.mkdir(self.report_dir)
        differential_results_file = open(os.path.join(self.report_dir, 'differential_results.json'), 'w')
        differential_results_file.write("The average precision differential for top " + str(top_k) + " is:" +
                                        str(avg_precision_differential))
        differential_results_file.write("\n The differential quality score for top " + str(top_k) + " is:" +
                                        str(quality_score))

    def find_errors_improvements(self, labels, probabilities, errors, improvements):
        """
        Identifies and returns errors and improvements, expected by report_results.
        Args:
            labels: A list containing the ground truth condition indices.
            probabilities: The list of probabilities which are the output of the visual-qa system.
            errors: A boolean value which determines if the data inconsistencies should be flagged.
            improvements: If true, the Q&A system improvements vs visual probabilities are stored.
        Returns:
            Errors and improvements made by the by-definition filter.
        """
        by_definition_errors = []
        by_definition_improvements = []
        test_matrix_diff_zeros = []
        test_matrix_diff_twos = []
        rank_change = []
        by_definition_df = self.visual_qa.general.knowledge
        filter_groups = self.visual_qa.general.create_filter_groups()
        by_definition_improvements.append(['case id', 'visual rank', 'visual-qa rank'])
        for index in range(len(labels)):
            # Rank in ascending order
            probs_sorted = sorted(self.prior_probabilities[index])
            filtered_sorted = sorted(probabilities[index])
            visual_gt_rank = 132 - np.searchsorted(probs_sorted, self.prior_probabilities[index][labels[index]])
            by_def_gt_rank = 132 - np.searchsorted(filtered_sorted, probabilities[index][labels[index]])
            rank_change.append(by_def_gt_rank - visual_gt_rank)
            if visual_gt_rank < by_def_gt_rank and errors:
                by_definition_errors.append(self.filtered_qa_data[index]['case_id'])
                test_matrix_diff_zeros, test_matrix_diff_twos = self.find_definition_errors(by_definition_df, index,
                                                                                            filter_groups,
                                                                                            test_matrix_diff_zeros,
                                                                                            test_matrix_diff_twos)

            if visual_gt_rank > by_def_gt_rank and improvements:
                by_definition_improvements.append(
                    (self.filtered_qa_data[index]['case_id'], visual_gt_rank, by_def_gt_rank)
                )

        test_matrix_diff_zeros.insert(0, ['case_id', 'diagnosis_id', 'diagnosis_name', 'attribute'])

        test_matrix_diff_twos.insert(0, ['case_id', 'diagnosis_id', 'diagnosis_name', 'attribute'])

        return by_definition_errors, test_matrix_diff_zeros, test_matrix_diff_twos, by_definition_improvements, rank_change

    def find_definition_errors(self, by_definition_df, sample_index, filter_groups, test_matrix_diff_zeros,
                               test_matrix_diff_twos):
        """
        Finds all the errors made by the by definition filter.
        Args:
            by_definition_df: The by-definition matrix of must, may and never for every condition and attribute.
            sample_index: The index of the current test case.
            filter_groups: A dictionary with the all evidence attribute keys and values.
            test_matrix_diff_zeros: A list with all errors made because of 0 in the by definition matrix.
            test_matrix_diff_twos: A list with all errors made because of 2 in the by definition matrix.
        Returns:
             A list of errors made by inclusion and exclusion.
        """
        for key, value in self.filtered_qa_data[sample_index]['evidence'].items():
            if key != 'visual_probabilities':
                # row in the data-frame we are interested in.
                df_row = self.filtered_qa_data[sample_index]['diagnosis_id']
                if self.mode != "inclusion":
                    def_matrix_zeros_error = self.check_exclusion_errors(by_definition_df, sample_index, df_row, key,
                                                                         value)
                    test_matrix_diff_zeros.extend(def_matrix_zeros_error)

                if self.mode != "exclusion":
                    def_matrix_two_error = self.check_inclusion_errors(by_definition_df, sample_index, df_row, key,
                                                                       value, filter_groups)
                    test_matrix_diff_twos.extend(def_matrix_two_error)

        return test_matrix_diff_zeros, test_matrix_diff_twos

    def check_exclusion_errors(self, by_definition_df, sample_index, df_row, key, value):
        """
        Checks if the error occurs due to a zero in the by definition matrix.
        Args:
            by_definition_df: The by-definition matrix of must, may and never for every condition and attribute.
            sample_index: The index of the current test case.
            df_row: The index of the of the condition in the data frame.
            key: The attribute in evidence.
            value: The value of the attribute in the in the evidence.
        Returns:
            A list of errors made due to a zero in the by definition matrix.
        """
        def_matrix_zeros_error = []
        if self.valid_evidence_data[key]['multiple_answer'] and isinstance(value, list):
            for val in value:
                col_name = ''.join([str(key), ":", str(val)])
                if by_definition_df.at[df_row, col_name] == 0:
                    def_matrix_zeros_error.append([self.filtered_qa_data[sample_index]['case_id'],
                                                   self.filtered_qa_data[sample_index]['diagnosis_id'],
                                                   self.filtered_qa_data[sample_index]['diagnosis_name'], col_name])
        else:
            if isinstance(value, list):
                value = value[0]
            col_name = ''.join([str(key), ":", str(value)])
            if by_definition_df.at[df_row, col_name] == 0:
                def_matrix_zeros_error.append([self.filtered_qa_data[sample_index]['case_id'],
                                               self.filtered_qa_data[sample_index]['diagnosis_id'],
                                               self.filtered_qa_data[sample_index]['diagnosis_name'], col_name])
        return def_matrix_zeros_error

    def check_inclusion_errors(self, by_definition_df, sample_index, df_row, key, value, filter_groups):
        """
        Checks if the error occurs due to a two in the by definition matrix.
        Args:
            by_definition_df: The by-definition matrix of must, may and never for every condition and attribute.
            i: The index of the current test case.
            df_row: The index of the of the condition in the data frame.
            key: The attribute in evidence.
            value: The value of the attribute in the in the evidence.
            filter_groups: A dictionary with the all evidence attribute keys and values.
        Returns:
            A list of errors made due to a two in the by definition matrix.
        """
        def_matrix_two_error = []
        found_inclusion = False
        if self.valid_evidence_data[key]['multiple_answer'] and isinstance(value, list):
            for val in value:
                col_name = ''.join([str(key), ":", str(val)])
                if by_definition_df.at[df_row, col_name] == 2:
                    found_inclusion = True
        else:
            if isinstance(value, list):
                value = value[0]
            col_name = ''.join([str(key), ":", str(value)])
            if by_definition_df.at[df_row, col_name] == 2:
                found_inclusion = True
        if not found_inclusion:
            for col_name in filter_groups[key]:
                if by_definition_df.at[df_row, col_name] == 2:
                    def_matrix_two_error.append([self.filtered_qa_data[sample_index]['case_id'],
                                                 self.filtered_qa_data[sample_index]['diagnosis_id'],
                                                 self.filtered_qa_data[sample_index]['diagnosis_name'],
                                                 col_name])
        return def_matrix_two_error

    def find_top_k_questions(self, top_k, strict_mode, attribute_list=None, chosen_attributes=[]):
        """
        Finds the individual questions which improves the chosen metric the most in a non sequential manner. This is
        computed by adding a single attribute  at a time to the evidence at a time along with the chosen_attributes and
        visual probabilities, then finds the attribute(s) with best accuracy or quality score. This process is repeated
        for each attribute individually from attribute_list.
        Args:
            top_k: Number of top questions to be returned.
            strict_mode: If true, checks that all evidence value can be possible in the matrix. If false,
                        checks if the any evidence for a particular is true.
            differential: If true considers quality score, else considers accuracy.
            ranked_differential: If true quality score is computed for ranked differential.
            attribute_list: A list of attributes among which top-k questions are found.
            chosen_attributes: A list of attributes among which the most differentiable attribute is found.
        Returns: A list of the the top-k questions.
        """
        if not attribute_list:
            attribute_list = list(self.visual_qa.general.create_filter_groups().keys())
        top_k_score_list = []
        for attribute in attribute_list:
            self.qa_data = QAData(self.qa_data_json)
            self.filtered_qa_data = self.qa_data.filter_diagnosis_ids(list(self.visual_qa.general.diagnoses))
            for i in range(len(self.filtered_qa_data)):
                temp_dict = {}
                for key, value in self.filtered_qa_data[i]['evidence'].items():
                    if key == attribute or key == 'visual_probabilities' or key in chosen_attributes:
                        temp_dict[key] = value
                self.filtered_qa_data[i]['evidence'] = temp_dict

            results = self.evaluate(self.prior_probabilities, strict_mode, report=False)
            avg_acc_result = results['average']['accuracy'][-1]
            top_k_score_list.append(avg_acc_result)

        indices = list(np.argsort(np.array(top_k_score_list)))[-top_k:]
        self.top_questions = [list(attribute_list)[i] for i in indices]
        self.qa_data = QAData(self.qa_data_json)
        self.filtered_qa_data = self.qa_data.filter_diagnosis_ids(list(self.visual_qa.general.diagnoses))
        return self.top_questions

    def find_top_k_questions_sequential(self, top_k, strict_mode, attribute_list=None):
        """
        Finds the group of questions which improves the chosen metric the most in a sequential manner. This is computed
        by adding a single attribute  at a time to the evidence at a time along with the previously found
        self.top_questions_sequential and visual probabilities, then finds the attribute(s) with best accuracy or
        quality score, self.top_questions_sequential at the i+1th iteration contains i attributes found one after the
        other.
        Args:
            top_k: Number of top questions to be returned.
            strict_mode: If true, checks that all evidence value can be possible in the matrix. If false,
                        checks if the any evidence for a particular is true.
            differential: If true considers quality score, else considers accuracy.
            ranked_differential: If true quality score is computed for ranked differential.
            attribute_list: A list of attributes among which top-k questions are found.
        Returns: A list of the the top-k questions.
        """
        if not attribute_list:
            attribute_list = list(self.visual_qa.general.create_filter_groups().keys())
        self.top_questions_sequential = []
        for top_i in range(top_k):
            remaining_attributes = [attribute for attribute in attribute_list if attribute not in
                                    self.top_questions_sequential]
            top_i_questions = self.find_top_k_questions(top_k, strict_mode,
                                                        attribute_list=remaining_attributes,
                                                        chosen_attributes=self.top_questions_sequential)
            # Storing the best ith question
            self.top_questions_sequential.append(top_i_questions[0])
        return self.top_questions_sequential
