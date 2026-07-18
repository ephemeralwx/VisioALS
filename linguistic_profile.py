import re
import math
import requests
from collections import Counter


# common English contractions for detection
_CONTRACTIONS = re.compile(
    r"\b(?:i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|she's|it's|"
    r"we're|we've|we'll|we'd|they're|they've|they'll|they'd|"
    r"isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|"
    r"doesn't|don't|didn't|won't|wouldn't|can't|couldn't|shouldn't|"
    r"mustn't|shan't|let's|that's|who's|what's|here's|there's|"
    r"where's|how's|ain't|gonna|wanna|gotta|kinda|sorta)\b",
    re.IGNORECASE,
)

# common filler words / phrases
_FILLERS = [
    "honestly", "look", "I mean", "the thing is", "you know",
    "basically", "actually", "like", "well", "so", "right",
    "anyway", "literally", "obviously", "clearly", "seriously",
    "frankly", "to be honest", "at the end of the day",
    "the point is", "fair enough", "no worries",
]

# High-signal UK colloquialisms. The LLM analysis below handles language
# varieties generally; this small deterministic layer makes common British
# English evidence survive even if the subjective-analysis request fails.
_BRITISH_MARKERS = [
    "mate", "cheers", "knackered", "shattered", "cuppa", "innit",
    "faff", "bloke", "sarnie", "kip", "ropey", "dodgy", "chuffed",
    "gutted", "muppet", "plonker", "git", "arse", "bollocks", "pub",
    "proper", "bang on", "crack on", "sack off", "do one",
    "taking the piss", "take the piss", "doing my head in",
    "done my head in", "happy days", "good egg", "sorted",
]


class LinguisticProfileExtractor:
    """Extracts a linguistic profile from a patient's text corpus."""

    def __init__(self, corpus: list[str], api_url: str):
        self._corpus = corpus
        self._api_url = api_url.rstrip("/")
        self._nlp = None

    def _load_spacy(self):
        if self._nlp is not None:
            return
        import spacy
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self._nlp = spacy.load("en_core_web_sm")

    def extract(self, progress_callback=None) -> dict:
        """Run all analyses and return the full linguistic profile dict."""
        self._load_spacy()

        if progress_callback:
            progress_callback(5)

        vocab = self._compute_vocabulary_metrics()
        if progress_callback:
            progress_callback(20)

        structure = self._compute_structural_patterns()
        if progress_callback:
            progress_callback(35)

        register = self._compute_register_tone()
        if progress_callback:
            progress_callback(50)

        phrases = self._extract_signature_phrases()
        regional = self._extract_regional_language()
        if progress_callback:
            progress_callback(65)

        # subjective analysis via LLM — may fail gracefully
        subjective = self._compute_subjective_analysis()
        if progress_callback:
            progress_callback(85)

        summary = self._build_summary(
            vocab, structure, register, subjective, phrases, regional
        )
        if progress_callback:
            progress_callback(100)

        return {
            "vocabulary": vocab,
            "structure": structure,
            "register": register,
            "subjective": subjective,
            "signature_phrases": phrases,
            "regional_language": regional,
            "summary": summary,
        }

    # ── vocabulary metrics ──────────────────────────────────────────

    def _compute_vocabulary_metrics(self) -> dict:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        all_tokens: list[str] = []
        sentence_lengths: list[int] = []

        for text in self._corpus:
            doc = self._nlp(text)
            for sent in doc.sents:
                words = [t.text for t in sent if not t.is_punct and not t.is_space]
                if words:
                    sentence_lengths.append(len(words))
                    all_tokens.extend(w.lower() for w in words)

        avg_sent_len = sum(sentence_lengths) / max(len(sentence_lengths), 1)

        # sliding window type-token ratio (window=100 tokens)
        window = 100
        if len(all_tokens) <= window:
            ttr = len(set(all_tokens)) / max(len(all_tokens), 1)
        else:
            ratios = []
            for i in range(0, len(all_tokens) - window + 1, window // 2):
                chunk = all_tokens[i : i + window]
                ratios.append(len(set(chunk)) / len(chunk))
            ttr = sum(ratios) / len(ratios)

        # TF-IDF distinctive words — treat each corpus entry as a "document"
        # and use English stop words to filter out common words
        distinctive_words: list[str] = []
        distinctive_phrases: list[str] = []

        if len(self._corpus) >= 2:
            # unigrams
            vec_uni = TfidfVectorizer(
                stop_words="english", max_features=500,
                ngram_range=(1, 1), min_df=2, max_df=0.9,
            )
            try:
                tfidf = vec_uni.fit_transform(self._corpus)
                scores = np.asarray(tfidf.mean(axis=0)).flatten()
                feature_names = vec_uni.get_feature_names_out()
                top_idx = scores.argsort()[::-1][:30]
                distinctive_words = [feature_names[i] for i in top_idx if scores[i] > 0]
            except ValueError:
                pass

            # bigrams / trigrams for distinctive phrases
            vec_ngram = TfidfVectorizer(
                stop_words="english", max_features=200,
                ngram_range=(2, 3), min_df=2, max_df=0.9,
            )
            try:
                tfidf_ng = vec_ngram.fit_transform(self._corpus)
                scores_ng = np.asarray(tfidf_ng.mean(axis=0)).flatten()
                feature_names_ng = vec_ngram.get_feature_names_out()
                top_ng = scores_ng.argsort()[::-1][:15]
                distinctive_phrases = [feature_names_ng[i] for i in top_ng if scores_ng[i] > 0]
            except ValueError:
                pass

        return {
            "avg_sentence_length": round(avg_sent_len, 1),
            "type_token_ratio": round(ttr, 3),
            "distinctive_words": distinctive_words[:30],
            "distinctive_phrases": distinctive_phrases[:15],
        }

    # ── structural patterns ─────────────────────────────────────────

    def _compute_structural_patterns(self) -> dict:
        type_counts = Counter()
        response_lengths: list[int] = []

        for text in self._corpus:
            doc = self._nlp(text)
            words = [t for t in doc if not t.is_punct and not t.is_space]
            response_lengths.append(len(words))

            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                tokens = [t for t in sent if not t.is_space]
                non_punct = [t for t in tokens if not t.is_punct]

                if sent_text.endswith("?"):
                    type_counts["question"] += 1
                elif len(non_punct) <= 3 and not any(t.dep_ == "ROOT" and t.pos_ == "VERB" for t in sent):
                    type_counts["fragment"] += 1
                elif any(t.dep_ == "ROOT" and t.tag_ in ("VB", "VBP") and t == sent[0] for t in sent if not t.is_punct):
                    type_counts["imperative"] += 1
                else:
                    type_counts["declarative"] += 1

        total = max(sum(type_counts.values()), 1)
        distribution = {k: round(v / total, 2) for k, v in type_counts.items()}
        avg_resp_len = sum(response_lengths) / max(len(response_lengths), 1)

        return {
            "avg_response_length": round(avg_resp_len, 1),
            "sentence_type_distribution": distribution,
        }

    # ── register / tone ─────────────────────────────────────────────

    def _compute_register_tone(self) -> dict:
        total_words = 0
        total_chars = 0
        contraction_count = 0
        passive_count = 0
        total_sentences = 0
        informal_marker_count = 0

        for text in self._corpus:
            doc = self._nlp(text)
            words = [t for t in doc if not t.is_punct and not t.is_space]
            total_words += len(words)
            total_chars += sum(len(t.text) for t in words)

            # contractions
            contraction_count += len(_CONTRACTIONS.findall(text))
            text_lower = text.lower()
            informal_marker_count += sum(
                len(re.findall(rf"(?<!\w){re.escape(marker)}(?!\w)", text_lower))
                for marker in _BRITISH_MARKERS
            )

            for sent in doc.sents:
                total_sentences += 1
                # passive voice: look for nsubjpass dependency
                if any(t.dep_ == "nsubjpass" for t in sent):
                    passive_count += 1

        avg_word_len = total_chars / max(total_words, 1)
        contraction_rate = contraction_count / max(total_words, 1)
        passive_ratio = passive_count / max(total_sentences, 1)
        informal_marker_rate = informal_marker_count / max(total_words, 1)

        # formality heuristic: higher = more formal
        # low contractions, longer words, more passive = more formal
        formality = (
            (1 - min(contraction_rate * 20, 1)) * 0.35 +
            min(avg_word_len / 7, 1) * 0.25 +
            min(passive_ratio * 5, 1) * 0.15 +
            (1 - min(informal_marker_rate * 12, 1)) * 0.25
        )

        return {
            "formality_score": round(formality, 2),
            "contraction_rate": round(contraction_rate, 3),
            "passive_voice_ratio": round(passive_ratio, 3),
            "avg_word_length": round(avg_word_len, 1),
            "informal_marker_rate": round(informal_marker_rate, 3),
        }

    # ── signature phrases ───────────────────────────────────────────

    def _extract_signature_phrases(self) -> dict:
        filler_counts = Counter()
        openers = Counter()
        closers = Counter()
        # Count in how many independent samples an n-gram appears. Repetition
        # within one essay reflects that essay's topic, not a general-purpose
        # catchphrase the patient is likely to use in conversation.
        ngram_document_counts = Counter()

        for text in self._corpus:
            text_lower = text.lower()

            # filler detection
            for filler in _FILLERS:
                count = text_lower.count(filler.lower())
                if count > 0:
                    filler_counts[filler] += count

            # openers (first 3 words) and closers (last 3 words)
            doc = self._nlp(text)
            words = [t.text for t in doc if not t.is_space]
            if len(words) >= 2:
                opener = " ".join(words[:min(3, len(words))])
                openers[opener] += 1
                closer = " ".join(words[-min(3, len(words)):])
                closers[closer] += 1

            # n-grams (2-4 words) for catchphrases
            tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]
            sample_ngrams = set()
            for n in (2, 3, 4):
                for i in range(len(tokens) - n + 1):
                    sample_ngrams.add(" ".join(tokens[i : i + n]))
            ngram_document_counts.update(sorted(sample_ngrams))

        # A catchphrase must occur in at least 3 separate samples.
        catchphrases = [
            phrase for phrase, count in ngram_document_counts.most_common(50)
            if count >= 3
        ][:10]

        return {
            "fillers": [f for f, _ in filler_counts.most_common(10)],
            "openers": [o for o, _ in openers.most_common(8)],
            "closers": [c for c, _ in closers.most_common(8)],
            "catchphrases": catchphrases,
        }

    def _extract_regional_language(self) -> dict:
        """Collect repeated, high-signal regional vocabulary."""
        counts = Counter()
        document_counts = Counter()

        for text in self._corpus:
            text_lower = text.lower()
            found_in_sample = set()
            for marker in _BRITISH_MARKERS:
                count = len(re.findall(
                    rf"(?<!\w){re.escape(marker)}(?!\w)", text_lower
                ))
                if count:
                    counts[marker] += count
                    found_in_sample.add(marker)
            document_counts.update(found_in_sample)

        # Prefer language repeated across independent samples, then fall back
        # to strong one-off evidence when a corpus is small.
        ranked = sorted(
            counts,
            key=lambda marker: (
                document_counts[marker] >= 2,
                document_counts[marker],
                counts[marker],
            ),
            reverse=True,
        )
        markers = ranked[:12]
        detected_variety = (
            "colloquial British English" if len(markers) >= 3 else "unknown"
        )
        return {
            "detected_variety": detected_variety,
            "markers": markers,
            "marker_counts": {marker: counts[marker] for marker in markers},
        }

    # ── subjective analysis (LLM call) ─────────────────────────────

    def _select_representative_samples(self, n: int = 50) -> list[str]:
        """Select diverse representative samples from the corpus."""
        if len(self._corpus) <= n:
            return list(self._corpus)

        # cluster by TF-IDF and pick from each cluster
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        import numpy as np

        vec = TfidfVectorizer(max_features=200, stop_words="english")
        try:
            X = vec.fit_transform(self._corpus)
        except ValueError:
            # fallback: evenly spaced samples
            step = max(len(self._corpus) // n, 1)
            return [self._corpus[i] for i in range(0, len(self._corpus), step)][:n]

        n_clusters = min(n, len(self._corpus))
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
        labels = km.fit_predict(X)

        # pick the sample closest to each cluster center
        selected = []
        for c in range(n_clusters):
            indices = [i for i, l in enumerate(labels) if l == c]
            if not indices:
                continue
            center = km.cluster_centers_[c]
            dists = [(i, float(np.linalg.norm(X[i].toarray() - center))) for i in indices]
            dists.sort(key=lambda x: x[1])
            selected.append(self._corpus[dists[0][0]])

        return selected[:n]

    def _compute_subjective_analysis(self) -> dict:
        """Call the LLM to analyze subjective style traits."""
        samples = self._select_representative_samples(50)
        if not samples:
            return {
                "humor_style": "unknown",
                "tone_description": "unknown",
                "emotional_valence": "neutral",
                "personality_notes": "",
                "language_variety": "unknown",
                "slang_and_regionalisms": [],
            }

        try:
            r = requests.post(
                f"{self._api_url}/analyze-style",
                json={"sample_texts": samples},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            return {
                "humor_style": data.get("humor_style", "unknown"),
                "tone_description": data.get("tone_description", "unknown"),
                "emotional_valence": data.get("emotional_valence", "neutral"),
                "personality_notes": data.get("personality_notes", ""),
                "language_variety": data.get("language_variety", "unknown"),
                "slang_and_regionalisms": data.get("slang_and_regionalisms", []),
            }
        except Exception as e:
            print(f"subjective analysis failed: {e}")
            return {
                "humor_style": "unknown",
                "tone_description": "unknown",
                "emotional_valence": "neutral",
                "personality_notes": "",
                "language_variety": "unknown",
                "slang_and_regionalisms": [],
            }

    # ── summary generation ──────────────────────────────────────────

    def _build_summary(self, vocab: dict, structure: dict, register: dict,
                       subjective: dict, phrases: dict,
                       regional: dict | None = None) -> str:
        parts = []

        # sentence length
        parts.append(f"Sentences average {vocab['avg_sentence_length']} words")

        # formality
        score = register["formality_score"]
        if score < 0.35:
            parts.append("very informal register")
        elif score < 0.5:
            parts.append("informal register")
        elif score < 0.65:
            parts.append("moderate formality")
        else:
            parts.append("formal register")

        # contractions
        if register["contraction_rate"] > 0.05:
            parts.append("frequently uses contractions")

        # tone
        if subjective["tone_description"] != "unknown":
            parts.append(f"tone is {subjective['tone_description']}")

        # humor
        if subjective["humor_style"] not in ("unknown", "none", ""):
            parts.append(f"{subjective['humor_style']} humor")

        # Explicitly preserve language variety and slang in the final string
        # that is actually sent to response generation.
        language_variety = subjective.get("language_variety", "unknown")
        if language_variety not in ("unknown", "none", ""):
            parts.append(f"language variety is {language_variety}")

        regional = regional or {}
        detected_variety = regional.get("detected_variety", "unknown")
        if (detected_variety not in ("unknown", "none", "") and
                language_variety in ("unknown", "none", "")):
            parts.append(f"language variety is {detected_variety}")

        slang = list(subjective.get("slang_and_regionalisms", []))
        for marker in regional.get("markers", []):
            if marker.lower() not in {str(item).lower() for item in slang}:
                slang.append(marker)
        if slang:
            quoted = "', '".join(str(item) for item in slang[:10])
            parts.append(
                f"naturally uses regional slang including '{quoted}'"
            )

        # emotional valence
        if subjective["emotional_valence"] not in ("unknown", "neutral"):
            parts.append(f"tends toward {subjective['emotional_valence']} framing")

        # distinctive words
        if vocab["distinctive_words"]:
            top3 = vocab["distinctive_words"][:3]
            words_str = "', '".join(top3)
            parts.append(f"distinctive vocabulary includes '{words_str}'")

        # signature phrases
        if phrases["fillers"]:
            top2 = phrases["fillers"][:2]
            fillers_str = "', '".join(top2)
            parts.append(f"common fillers: '{fillers_str}'")

        if phrases["catchphrases"]:
            parts.append("catchphrase: '" + phrases["catchphrases"][0] + "'")

        # sentence structure
        dist = structure.get("sentence_type_distribution", {})
        dominant = max(dist, key=dist.get) if dist else None
        if dominant and dominant != "declarative":
            parts.append(f"notable use of {dominant} sentences")

        return ". ".join(parts) + "."
