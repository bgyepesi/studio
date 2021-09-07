from studio.evaluation.keras.data import QAData


def test_qa_data(test_qa_data_json_path):
    qa_data = QAData(test_qa_data_json_path)
    assert len(qa_data.samples) == 3


def test_filter_diagnosis_ids(test_qa_data_json_path):
    qa_data = QAData(test_qa_data_json_path)
    diagnosis_ids = ['AIP:0000733', 'AIP:0000734', 'AIP:0000303']
    filtered_tests = qa_data.filter_diagnosis_ids(diagnosis_ids)
    assert len(filtered_tests) == 2
    assert len(filtered_tests[1]['differential']) == 1
