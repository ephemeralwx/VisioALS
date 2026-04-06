import os
import json
import csv
from patient_data import PatientDataManager


def test_directory_creation(tmp_patient_dir):
    pm = PatientDataManager("alice", base_dir=tmp_patient_dir)
    assert os.path.isdir(pm.patient_dir)
    assert os.path.isdir(pm.corpus_dir)
    assert os.path.isdir(pm.embeddings_dir)


def test_load_corpus_txt(patient_with_corpus):
    texts = patient_with_corpus.load_corpus()
    assert len(texts) == 30
    assert "Honestly" in texts[0]


def test_load_corpus_json(tmp_patient_dir):
    pm = PatientDataManager("json_patient", base_dir=tmp_patient_dir)
    data = ["Hello there", "How are you", "I'm fine"]
    with open(os.path.join(pm.corpus_dir, "data.json"), "w") as f:
        json.dump(data, f)
    texts = pm.load_corpus()
    assert texts == data


def test_load_corpus_csv(tmp_patient_dir):
    pm = PatientDataManager("csv_patient", base_dir=tmp_patient_dir)
    csv_path = os.path.join(pm.corpus_dir, "data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "date"])
        writer.writeheader()
        writer.writerow({"text": "Row one", "date": "2025-01-01"})
        writer.writerow({"text": "Row two", "date": "2025-01-02"})
    texts = pm.load_corpus()
    assert texts == ["Row one", "Row two"]


def test_load_corpus_csv_no_text_column(tmp_patient_dir):
    pm = PatientDataManager("csv_first_col", base_dir=tmp_patient_dir)
    csv_path = os.path.join(pm.corpus_dir, "data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["message", "id"])
        writer.writeheader()
        writer.writerow({"message": "First", "id": "1"})
        writer.writerow({"message": "Second", "id": "2"})
    texts = pm.load_corpus()
    assert texts == ["First", "Second"]


def test_linguistic_profile_roundtrip(tmp_patient_dir):
    pm = PatientDataManager("profile_test", base_dir=tmp_patient_dir)
    assert pm.load_linguistic_profile() is None

    profile = {"vocabulary": {"avg_sentence_length": 10}, "summary": "test"}
    pm.save_linguistic_profile(profile)

    loaded = pm.load_linguistic_profile()
    assert loaded["summary"] == "test"
    assert loaded["vocabulary"]["avg_sentence_length"] == 10


def test_preference_profile_roundtrip(tmp_patient_dir):
    pm = PatientDataManager("pref_test", base_dir=tmp_patient_dir)
    assert pm.load_preference_profile() is None

    profile = {"rules": ["Avoids sentimental language", "Prefers short answers"]}
    pm.save_preference_profile(profile)

    loaded = pm.load_preference_profile()
    assert loaded["rules"] == ["Avoids sentimental language", "Prefers short answers"]


def test_preference_profile_delete(tmp_patient_dir):
    pm = PatientDataManager("pref_del", base_dir=tmp_patient_dir)
    pm.save_preference_profile({"rules": ["test"]})
    assert pm.load_preference_profile() is not None
    pm.delete_preference_profile()
    assert pm.load_preference_profile() is None


def test_interaction_logging(tmp_patient_dir):
    pm = PatientDataManager("log_test", base_dir=tmp_patient_dir)
    assert pm.interaction_count() == 0
    assert pm.load_interactions() == []

    pm.log_interaction({
        "question": "How are you?",
        "options_presented": ["Good", "Bad", "Okay", "Fine"],
        "selected": "Fine",
        "rejected": ["Good", "Bad", "Okay"],
        "rejection_round": 0,
    })

    assert pm.interaction_count() == 1
    entries = pm.load_interactions()
    assert len(entries) == 1
    assert entries[0]["selected"] == "Fine"
    assert "timestamp" in entries[0]


def test_interaction_logging_last_n(tmp_patient_dir):
    pm = PatientDataManager("log_n_test", base_dir=tmp_patient_dir)
    for i in range(10):
        pm.log_interaction({"question": f"q{i}", "selected": f"a{i}"})

    last_3 = pm.load_interactions(last_n=3)
    assert len(last_3) == 3
    assert last_3[0]["question"] == "q7"
    assert last_3[2]["question"] == "q9"


def test_list_patients(tmp_patient_dir):
    PatientDataManager("alice", base_dir=tmp_patient_dir)
    PatientDataManager("bob", base_dir=tmp_patient_dir)
    patients = PatientDataManager.list_patients(tmp_patient_dir)
    assert patients == ["alice", "bob"]
