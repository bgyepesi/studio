from studio.data.ontology import Ontology

""" Checks for AIP's Master Ontology """


def check_ontology(ontology, diagnosis_nodes):
    ontology.set_diagnosis_nodes(diagnosis_nodes)
    assert len(ontology.errors_report) == 0


def test_macroscopic_ontology(aip_ontology_manifest, aip_macroscopic_diagnosis_nodes_df):
    check_ontology(Ontology(aip_ontology_manifest), aip_macroscopic_diagnosis_nodes_df['id'].tolist())


def test_dermoscopic_ontology(aip_ontology_manifest, aip_dermoscopic_diagnosis_nodes_df):
    check_ontology(Ontology(aip_ontology_manifest), aip_dermoscopic_diagnosis_nodes_df['id'].tolist())
