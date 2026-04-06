import os
import json
import csv
import time


def _default_base_dir() -> str:
    base = os.environ.get("APPDATA", os.path.expanduser("~"))
    return os.path.join(base, "VisioALS", "patients")


class PatientDataManager:
    """Manages the patient data directory and all file I/O for a single patient."""

    def __init__(self, patient_name: str, base_dir: str | None = None):
        self.patient_name = patient_name
        self._base_dir = base_dir or _default_base_dir()
        self.ensure_directories()

    @property
    def patient_dir(self) -> str:
        return os.path.join(self._base_dir, self.patient_name)

    @property
    def corpus_dir(self) -> str:
        return os.path.join(self.patient_dir, "corpus")

    @property
    def embeddings_dir(self) -> str:
        return os.path.join(self.patient_dir, "embeddings")

    @property
    def linguistic_profile_path(self) -> str:
        return os.path.join(self.patient_dir, "linguistic_profile.json")

    @property
    def preference_profile_path(self) -> str:
        return os.path.join(self.patient_dir, "preference_profile.json")

    @property
    def interaction_log_path(self) -> str:
        return os.path.join(self.patient_dir, "interaction_log.jsonl")

    def ensure_directories(self) -> None:
        for d in (self.patient_dir, self.corpus_dir, self.embeddings_dir):
            os.makedirs(d, exist_ok=True)

    # ── corpus ──────────────────────────────────────────────────────

    def load_corpus(self) -> list[str]:
        """Load all text snippets from the corpus directory.

        Supports:
        - .txt files: each file is one snippet
        - .json files: expects a JSON array of strings
        - .csv files: reads 'text' column, or first column if 'text' not found
        """
        texts: list[str] = []
        if not os.path.isdir(self.corpus_dir):
            return texts

        for fname in sorted(os.listdir(self.corpus_dir)):
            fpath = os.path.join(self.corpus_dir, fname)
            if not os.path.isfile(fpath):
                continue

            ext = os.path.splitext(fname)[1].lower()

            if ext == ".txt":
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    texts.append(content)

            elif ext == ".json":
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    texts.extend(str(item) for item in data if str(item).strip())

            elif ext == ".csv":
                with open(fpath, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    fields = reader.fieldnames or []
                    col = "text" if "text" in fields else (fields[0] if fields else None)
                    if col:
                        for row in reader:
                            val = (row.get(col) or "").strip()
                            if val:
                                texts.append(val)

        return texts

    # ── linguistic profile ──────────────────────────────────────────

    def load_linguistic_profile(self) -> dict | None:
        if not os.path.exists(self.linguistic_profile_path):
            return None
        with open(self.linguistic_profile_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_linguistic_profile(self, profile: dict) -> None:
        with open(self.linguistic_profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    # ── preference profile ──────────────────────────────────────────

    def load_preference_profile(self) -> dict | None:
        if not os.path.exists(self.preference_profile_path):
            return None
        with open(self.preference_profile_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_preference_profile(self, profile: dict) -> None:
        with open(self.preference_profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    def delete_preference_profile(self) -> None:
        if os.path.exists(self.preference_profile_path):
            os.remove(self.preference_profile_path)

    # ── interaction log ─────────────────────────────────────────────

    def log_interaction(self, entry: dict) -> None:
        """Append a single interaction entry to the JSONL log.

        Expected fields: timestamp, question, options_presented, selected,
        rejected, rejection_round.
        """
        if "timestamp" not in entry:
            entry["timestamp"] = time.time()
        with open(self.interaction_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_interactions(self, last_n: int = 100) -> list[dict]:
        """Read the last N interaction entries from the JSONL log."""
        if not os.path.exists(self.interaction_log_path):
            return []
        lines: list[str] = []
        with open(self.interaction_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        # take only the last N
        lines = lines[-last_n:]
        entries = []
        for line in lines:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def interaction_count(self) -> int:
        """Return total number of logged interactions."""
        if not os.path.exists(self.interaction_log_path):
            return 0
        count = 0
        with open(self.interaction_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    # ── patient listing ─────────────────────────────────────────────

    @staticmethod
    def list_patients(base_dir: str | None = None) -> list[str]:
        d = base_dir or _default_base_dir()
        if not os.path.isdir(d):
            return []
        return sorted(
            name for name in os.listdir(d)
            if os.path.isdir(os.path.join(d, name))
        )
