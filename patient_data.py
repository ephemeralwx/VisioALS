import os
import json
import csv
import re
import hashlib
import shutil
import time
import wave


def _default_base_dir() -> str:
    # appdata doesn't exist on linux/mac, fallback to home
    base = os.environ.get("APPDATA", os.path.expanduser("~"))
    return os.path.join(base, "VisioALS", "patients")


class PatientDataManager:

    @staticmethod
    def list_patient_names(base_dir: str | None = None) -> list[str]:
        """Return patients that have a saved linguistic or voice profile."""
        root = base_dir or _default_base_dir()
        if not os.path.isdir(root):
            return []

        names: list[str] = []
        for name in os.listdir(root):
            patient_dir = os.path.join(root, name)
            if not os.path.isdir(patient_dir):
                continue
            has_profile = any(
                os.path.isfile(os.path.join(patient_dir, filename))
                for filename in ("linguistic_profile.json", "voice_profile.json")
            )
            if has_profile:
                names.append(name)
        return sorted(names, key=str.casefold)

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
    def media_dir(self) -> str:
        return os.path.join(self.patient_dir, "media")

    @property
    def linguistic_profile_path(self) -> str:
        return os.path.join(self.patient_dir, "linguistic_profile.json")

    @property
    def preference_profile_path(self) -> str:
        return os.path.join(self.patient_dir, "preference_profile.json")

    @property
    def interaction_log_path(self) -> str:
        return os.path.join(self.patient_dir, "interaction_log.jsonl")

    @property
    def voice_profile_path(self) -> str:
        return os.path.join(self.patient_dir, "voice_profile.json")

    def ensure_directories(self) -> None:
        for d in (
            self.patient_dir,
            self.corpus_dir,
            self.embeddings_dir,
            self.media_dir,
        ):
            os.makedirs(d, exist_ok=True)

    def add_media_files(self, paths: list[str]) -> list[str]:
        """Copy imported audio/video into this patient's private media folder."""
        copied: list[str] = []
        for source in paths:
            if not os.path.isfile(source):
                continue
            base = os.path.basename(source)
            stem, ext = os.path.splitext(base)
            destination = os.path.join(self.media_dir, base)
            suffix = 2
            while os.path.exists(destination):
                destination = os.path.join(self.media_dir, f"{stem}_{suffix}{ext}")
                suffix += 1
            shutil.copy2(source, destination)
            copied.append(destination)
        return copied

    def add_corpus_files(self, paths: list[str]) -> list[str]:
        """Copy text corpus files without deleting existing spoken transcripts."""
        copied: list[str] = []
        for source in paths:
            if not os.path.isfile(source):
                continue
            base = os.path.basename(source)
            stem, ext = os.path.splitext(base)
            destination = os.path.join(self.corpus_dir, base)
            suffix = 2
            while os.path.exists(destination):
                destination = os.path.join(self.corpus_dir, f"{stem}_{suffix}{ext}")
                suffix += 1
            shutil.copy2(source, destination)
            copied.append(destination)
        return copied

    def save_recording(self, audio, sample_rate: int = 16000) -> str:
        """Store an in-app microphone recording as a voice/corpus source."""
        filename = f"recording_{int(time.time() * 1000)}.wav"
        destination = os.path.join(self.media_dir, filename)
        with wave.open(destination, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return destination

    def list_media_files(self) -> list[str]:
        if not os.path.isdir(self.media_dir):
            return []
        return [
            os.path.join(self.media_dir, name)
            for name in sorted(os.listdir(self.media_dir))
            if os.path.isfile(os.path.join(self.media_dir, name))
        ]

    def save_media_transcript(self, media_path: str, transcript: str) -> str | None:
        """Save a media transcript into the text corpus used by profiling/RAG."""
        content = (transcript or "").strip()
        if not content:
            return None
        stem = os.path.splitext(os.path.basename(media_path))[0]
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
        digest = hashlib.sha1(os.path.basename(media_path).encode("utf-8")).hexdigest()[:8]
        destination = os.path.join(self.corpus_dir, f"spoken_{safe_stem}_{digest}.txt")
        with open(destination, "w", encoding="utf-8") as f:
            f.write(content)
        return destination

    def media_fingerprint(self) -> str:
        """Hash imported media so an unchanged voice clone can be reused."""
        digest = hashlib.sha256()
        for path in self.list_media_files():
            digest.update(os.path.basename(path).encode("utf-8"))
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

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
                    # The onboarding text box is commonly filled with copied
                    # messages separated by blank lines. Keep those as
                    # independent samples so document-frequency metrics,
                    # catchphrase detection, and semantic retrieval can learn
                    # from them. Imported text files and spoken transcripts
                    # retain their original document boundaries.
                    if fname == "pasted_text.txt":
                        samples = [
                            sample.strip()
                            for sample in re.split(r"\n\s*\n+", content)
                            if sample.strip()
                        ]
                        texts.extend(samples or [content])
                    else:
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

    def replace_corpus_with_text(self, text: str) -> None:
        """Replace the imported corpus with text pasted during onboarding."""
        content = text.strip()
        if not content:
            raise ValueError("Corpus text cannot be empty.")

        for fname in os.listdir(self.corpus_dir):
            fpath = os.path.join(self.corpus_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)

        with open(
            os.path.join(self.corpus_dir, "pasted_text.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(content)

    def add_pasted_text(self, text: str) -> str:
        """Add pasted writing without removing imported media transcripts/files."""
        content = text.strip()
        if not content:
            raise ValueError("Corpus text cannot be empty.")
        destination = os.path.join(self.corpus_dir, "pasted_text.txt")
        with open(destination, "w", encoding="utf-8") as f:
            f.write(content)
        return destination

    def load_linguistic_profile(self) -> dict | None:
        if not os.path.exists(self.linguistic_profile_path):
            return None
        with open(self.linguistic_profile_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ensure_ascii=False everywhere so we don't mangle non-english patient text
    def save_linguistic_profile(self, profile: dict) -> None:
        with open(self.linguistic_profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    def load_voice_profile(self) -> dict | None:
        if not os.path.exists(self.voice_profile_path):
            return None
        with open(self.voice_profile_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_voice_profile(self, profile: dict) -> None:
        with open(self.voice_profile_path, "w", encoding="utf-8") as f:
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
