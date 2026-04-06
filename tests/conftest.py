import os
import sys
import json
import shutil
import pytest

# make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SMALL_CORPUS = os.path.join(FIXTURES, "sample_corpus_small")
LARGE_CORPUS = os.path.join(FIXTURES, "sample_corpus_large")
INTERACTIONS_FILE = os.path.join(FIXTURES, "sample_interactions.jsonl")


@pytest.fixture
def tmp_patient_dir(tmp_path):
    """Provides a temporary base directory for patient data."""
    return str(tmp_path / "patients")


@pytest.fixture
def small_corpus_texts():
    """Load the small (~30) test corpus as a list of strings."""
    texts = []
    for fname in sorted(os.listdir(SMALL_CORPUS)):
        if fname.endswith(".txt"):
            with open(os.path.join(SMALL_CORPUS, fname), "r") as f:
                texts.append(f.read().strip())
    return texts


@pytest.fixture
def large_corpus_texts():
    """Load the large (200+) test corpus as a list of strings."""
    with open(os.path.join(LARGE_CORPUS, "corpus.json"), "r") as f:
        return json.load(f)


@pytest.fixture
def sample_interactions():
    """Load the sample interaction log entries."""
    entries = []
    with open(INTERACTIONS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


@pytest.fixture
def patient_with_corpus(tmp_patient_dir, small_corpus_texts):
    """Create a PatientDataManager with the small corpus copied in."""
    from patient_data import PatientDataManager
    pm = PatientDataManager("test_patient", base_dir=tmp_patient_dir)
    # copy small corpus files
    for fname in sorted(os.listdir(SMALL_CORPUS)):
        if fname.endswith(".txt"):
            shutil.copy2(os.path.join(SMALL_CORPUS, fname), pm.corpus_dir)
    return pm


@pytest.fixture
def patient_with_large_corpus(tmp_patient_dir, large_corpus_texts):
    """Create a PatientDataManager with the large corpus."""
    from patient_data import PatientDataManager
    pm = PatientDataManager("test_patient_large", base_dir=tmp_patient_dir)
    corpus_file = os.path.join(pm.corpus_dir, "corpus.json")
    with open(corpus_file, "w") as f:
        json.dump(large_corpus_texts, f)
    return pm
