"""Tests for preference learning (interaction logging + preference extraction)."""
import time
import pytest
from unittest.mock import patch, MagicMock
from patient_data import PatientDataManager


def test_interaction_log_persists_across_instances(tmp_patient_dir):
    """Log written by one instance should be readable by another."""
    pm1 = PatientDataManager("persist_test", base_dir=tmp_patient_dir)
    pm1.log_interaction({"question": "test", "selected": "yes"})
    pm1.log_interaction({"question": "test2", "selected": "no"})

    pm2 = PatientDataManager("persist_test", base_dir=tmp_patient_dir)
    entries = pm2.load_interactions()
    assert len(entries) == 2
    assert pm2.interaction_count() == 2


def test_interaction_count_matches_entries(tmp_patient_dir, sample_interactions):
    pm = PatientDataManager("count_test", base_dir=tmp_patient_dir)
    for entry in sample_interactions:
        pm.log_interaction(entry)
    assert pm.interaction_count() == len(sample_interactions)


def test_interactions_have_required_fields(sample_interactions):
    """Every sample interaction should have the schema fields."""
    for entry in sample_interactions:
        assert "question" in entry
        assert "options_presented" in entry
        assert "selected" in entry or entry.get("selected") is None
        assert "rejected" in entry
        assert "rejection_round" in entry


def test_preference_profile_structure(tmp_patient_dir):
    pm = PatientDataManager("pref_struct", base_dir=tmp_patient_dir)
    profile = {
        "rules": [
            "Avoids sentimental or emotionally effusive language",
            "Prefers direct, practical responses",
            "Tends toward brevity (under 5 words)",
        ],
        "last_updated": time.time(),
        "interaction_count": 50,
    }
    pm.save_preference_profile(profile)

    loaded = pm.load_preference_profile()
    assert len(loaded["rules"]) == 3
    assert loaded["interaction_count"] == 50
    assert "last_updated" in loaded


def test_clear_preferences_resets(tmp_patient_dir):
    pm = PatientDataManager("clear_test", base_dir=tmp_patient_dir)
    pm.save_preference_profile({"rules": ["test rule"]})
    assert pm.load_preference_profile() is not None

    pm.delete_preference_profile()
    assert pm.load_preference_profile() is None


def test_sample_interactions_show_clear_pattern(sample_interactions):
    """The sample data should show a clear pattern: patient consistently
    rejects sentimental/effusive options and picks direct/practical ones."""
    sentimental_keywords = ["wonderful", "blessed", "precious", "divine",
                            "magnificent", "heavenly", "treasure", "delight",
                            "glorious", "overflowing", "miracle", "bliss"]

    rejected_sentimental = 0
    selected_sentimental = 0
    total = len(sample_interactions)

    for entry in sample_interactions:
        selected = (entry.get("selected") or "").lower()
        rejected_texts = [r.lower() for r in entry.get("rejected", [])]

        for kw in sentimental_keywords:
            if any(kw in r for r in rejected_texts):
                rejected_sentimental += 1
                break

        for kw in sentimental_keywords:
            if kw in selected:
                selected_sentimental += 1
                break

    # patient should reject sentimental options in most interactions
    assert rejected_sentimental > total * 0.6, (
        f"Expected >60% of interactions to reject sentimental options, "
        f"got {rejected_sentimental}/{total}"
    )
    # patient should almost never select sentimental options
    assert selected_sentimental < total * 0.05, (
        f"Expected <5% of selections to be sentimental, "
        f"got {selected_sentimental}/{total}"
    )
