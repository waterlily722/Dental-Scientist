import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_MEMORY_FILENAME = "lab_memory.jsonl"
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def _memory_path(memory_dir: str) -> Path:
    path = Path(memory_dir)
    if path.suffix == ".jsonl":
        return path
    return path / DEFAULT_MEMORY_FILENAME


def load_memory_records(memory_dir: str) -> List[Dict[str, Any]]:
    path = _memory_path(memory_dir)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def retrieve_relevant_memories(
    memory_dir: str,
    *,
    query_text: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    records = load_memory_records(memory_dir)
    if not records:
        return []

    query_tokens = Counter(_tokenize(query_text))
    scored: list[tuple[int, Dict[str, Any]]] = []
    for record in records:
        haystack = " ".join(
            str(record.get(key, ""))
            for key in ["idea_name", "change_axis", "failure_reason", "lesson"]
        )
        record_tokens = Counter(_tokenize(haystack))
        score = sum(min(query_tokens[token], record_tokens[token]) for token in query_tokens)
        if score > 0:
            scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        return [record for _, record in scored[: max(1, min(limit, 5))]]
    return records[-max(1, min(limit, 5)) :]


def append_memory_record(memory_dir: str, record: Dict[str, Any]) -> Path:
    path = _memory_path(memory_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def format_memory_for_prompt(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "No relevant historical lab notes found."
    lines = []
    for idx, record in enumerate(records, start=1):
        lines.append(
            (
                f"{idx}. idea={record.get('idea_name', '')}; "
                f"axis={record.get('change_axis', '')}; "
                f"success={record.get('success', '')}; "
                f"failure_reason={record.get('failure_reason', '')}; "
                f"lesson={record.get('lesson', '')}"
            ).strip()
        )
    return "\n".join(lines)
