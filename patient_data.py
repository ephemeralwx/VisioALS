import os
import json
import csv
import time


def _default_base_dir() -> str:
    # appdata doesn't exist on linux/mac, fallback to home
    base = os.environ.get("APPDATA", os.path.expanduser("~"))
    return os.path.join(base, "VisioALS", "patients")


class PatientDataManager:

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

    def load_corpus(self) -> list[str]:
        texts: list[str] = []
        if not os.path.isdir(self.corpus_dir):
            return texts

        # sorted so embedding order stays deterministic across runs
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
                    # try "text" column first, otherwise just grab whatever the first col is
                    col = "text" if "text" in fields else (fields[0] if fields else None)
                    if col:
                        for row in reader:
                            val = (row.get(col) or "").strip()
                            if val:
                                texts.append(val)
        # print(f"loaded {len(texts)} snippets")
        return texts

    def load_linguistic_profile(self) -> dict | None:
        if not os.path.exists(self.linguistic_profile_path):
            return None
        with open(self.linguistic_profile_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ensure_ascii=False everywhere so we don't mangle non-english patient text
    def save_linguistic_profile(self, profile: dict) -> None:
        with open(self.linguistic_profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

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

    def log_interaction(self, entry: dict) -> None:
        if "timestamp" not in entry:
            entry["timestamp"] = time.time()
        # jsonl so we can just append without rewriting the whole file
        with open(self.interaction_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_interactions(self, last_n: int = 100) -> list[dict]:
        if not os.path.exists(self.interaction_log_path):
            return []
        # TODO: reads the whole file to get last N, kinda wasteful
        # but the log shouldnt get that big in practice so whatever
        lines: list[str] = []
        with open(self.interaction_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        lines = lines[-last_n:]
        entries = []
        for line in lines:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # corrupted line, just skip - dont blow up the whole load
                continue
        return entries

    def interaction_count(self) -> int:
        if not os.path.exists(self.interaction_log_path):
            return 0
        count = 0
        with open(self.interaction_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
        # return len(self.load_interactions(last_n=999999))

    @staticmethod
    def list_patients(base_dir: str | None = None) -> list[str]:
        d = base_dir or _default_base_dir()
        if not os.path.isdir(d):
            return []
        return sorted(
            name for name in os.listdir(d)
            if os.path.isdir(os.path.join(d, name))
        )
