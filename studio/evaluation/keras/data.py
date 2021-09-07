import json


class QAData(object):
    """
    This class processes the test set examples from a json file.
    A test case is formed by:
        `case_id`: Test case identifier.
        `class_name` : Clinical name of the diagnosis node.
        `diagnosis_id`: AIP id of the diagnosis node that this test case belongs to.
        `differential`: List of AIP diagnosis nodes identifiers related to the differential diagnosis.
        `evidence`: The clinical case properties that the Q&A system uses to refine the diagnosis. An evidence consists
        of a set of attributes and its values.
        `image_id`: Image filename for the test case.
    """

    def __init__(self, qa_data_json):
        with open(qa_data_json) as file:
            self.samples = json.load(file)

        self.evidence_keys = list(self.samples[0]['evidence'].keys())

    def __len__(self):
        return len(self.samples)

    def filter_diagnosis_ids(self, diagnosis_ids):
        """
        Returns the data samples covered by any of the `diagnosis_ids`.
        Args:
            diagnosis_ids: AIP id of the diagnosis node that this test case belongs to.
        Returns: The list of filtered test cases covered by `diagnosis_ids`.
        """
        self.filtered_samples = []
        for sample in self.samples:
            if sample['diagnosis_id'] in diagnosis_ids:
                diff_set = set(sample['differential'])
                if len(set(diagnosis_ids).intersection(diff_set)) < len(diff_set):
                    filtered_ddx = [ddx for ddx in sample['differential'] if ddx in diagnosis_ids]
                    sample['differential'] = filtered_ddx
                self.filtered_samples.append(sample)
        return self.filtered_samples
