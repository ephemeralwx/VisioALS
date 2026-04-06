"""Tests for linguistic profile extraction.

Unit tests use the small corpus. Integration tests use the large corpus
to verify TF-IDF and TTR produce meaningful values at scale.
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def extractor_small(small_corpus_texts):
    from linguistic_profile import LinguisticProfileExtractor
    return LinguisticProfileExtractor(small_corpus_texts, "http://fake-api")


@pytest.fixture
def extractor_large(large_corpus_texts):
    from linguistic_profile import LinguisticProfileExtractor
    return LinguisticProfileExtractor(large_corpus_texts, "http://fake-api")


class TestVocabularyMetrics:
    def test_avg_sentence_length_positive(self, extractor_small):
        extractor_small._load_spacy()
        metrics = extractor_small._compute_vocabulary_metrics()
        assert metrics["avg_sentence_length"] > 0

    def test_ttr_in_range(self, extractor_small):
        extractor_small._load_spacy()
        metrics = extractor_small._compute_vocabulary_metrics()
        assert 0 < metrics["type_token_ratio"] <= 1.0

    def test_large_corpus_ttr_reasonable(self, extractor_large):
        """With 200+ snippets, TTR should be meaningfully below 1.0."""
        extractor_large._load_spacy()
        metrics = extractor_large._compute_vocabulary_metrics()
        assert 0.3 < metrics["type_token_ratio"] < 0.95

    def test_large_corpus_distinctive_words(self, extractor_large):
        """With enough data, TF-IDF should find distinctive words."""
        extractor_large._load_spacy()
        metrics = extractor_large._compute_vocabulary_metrics()
        assert len(metrics["distinctive_words"]) > 0


class TestStructuralPatterns:
    def test_response_length_positive(self, extractor_small):
        extractor_small._load_spacy()
        structure = extractor_small._compute_structural_patterns()
        assert structure["avg_response_length"] > 0

    def test_sentence_types_present(self, extractor_small):
        extractor_small._load_spacy()
        structure = extractor_small._compute_structural_patterns()
        dist = structure["sentence_type_distribution"]
        assert sum(dist.values()) == pytest.approx(1.0, abs=0.05)


class TestRegisterTone:
    def test_formality_in_range(self, extractor_small):
        extractor_small._load_spacy()
        register = extractor_small._compute_register_tone()
        assert 0 <= register["formality_score"] <= 1.0

    def test_contractions_detected(self, extractor_small):
        """Our test corpus is informal and uses contractions."""
        extractor_small._load_spacy()
        register = extractor_small._compute_register_tone()
        assert register["contraction_rate"] > 0


class TestSignaturePhrases:
    def test_fillers_detected(self, extractor_small):
        extractor_small._load_spacy()
        phrases = extractor_small._extract_signature_phrases()
        # our test corpus uses "honestly", "I mean", "look", "the thing is"
        assert len(phrases["fillers"]) > 0
        found = {f.lower() for f in phrases["fillers"]}
        assert "honestly" in found or "I mean".lower() in found

    def test_openers_present(self, extractor_small):
        extractor_small._load_spacy()
        phrases = extractor_small._extract_signature_phrases()
        assert len(phrases["openers"]) > 0


class TestSubjectiveAnalysis:
    def test_returns_default_on_api_failure(self, extractor_small):
        """Should gracefully return defaults if the API call fails."""
        result = extractor_small._compute_subjective_analysis()
        # API at http://fake-api will fail, should return defaults
        assert result["humor_style"] == "unknown"

    def test_representative_samples_count(self, extractor_large):
        samples = extractor_large._select_representative_samples(50)
        assert len(samples) == 50


class TestFullExtraction:
    @patch("linguistic_profile.requests.post")
    def test_extract_returns_all_fields(self, mock_post, extractor_small):
        # mock the /analyze-style call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "humor_style": "dry",
            "tone_description": "direct and warm",
            "emotional_valence": "positive",
            "personality_notes": "practical, no-nonsense",
        }
        mock_post.return_value = mock_response

        profile = extractor_small.extract()

        assert "vocabulary" in profile
        assert "structure" in profile
        assert "register" in profile
        assert "subjective" in profile
        assert "signature_phrases" in profile
        assert "summary" in profile
        assert isinstance(profile["summary"], str)
        assert len(profile["summary"]) > 10
