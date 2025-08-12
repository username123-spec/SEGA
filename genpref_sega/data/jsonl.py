from typing import Iterator, Dict, Any, List, Optional
import json
def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            yield json.loads(line)
def load_pairs(path: str) -> List[Dict[str, str]]:
    return list(read_jsonl(path))
