from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


PREP_VERSION = "0.1.0-ambuda-dcs-passages"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _parse_dcs_file(
    path: Path,
    *,
    lipi: Any,
) -> List[Dict[str, Any]]:
    passages: List[Dict[str, Any]] = []

    current_id = None
    surfaces: List[str] = []
    lemmas: List[str] = []
    features: List[str] = []

    def flush() -> None:
        nonlocal current_id, surfaces, lemmas, features

        if not current_id or not surfaces:
            current_id = None
            surfaces = []
            lemmas = []
            features = []
            return

        slp1_text = " ".join(surfaces)
        slp1_lemmas = " ".join(lemmas)

        devanagari_text = lipi.transliterate(
            slp1_text,
            lipi.Scheme.Slp1,
            lipi.Scheme.Devanagari,
        )

        devanagari_lemmas = lipi.transliterate(
            slp1_lemmas,
            lipi.Scheme.Slp1,
            lipi.Scheme.Devanagari,
        )

        passages.append(
            {
                "passage_id": current_id,
                "source_file": path.name,
                "surface_slp1": slp1_text,
                "surface_devanagari": devanagari_text,
                "lemma_slp1": slp1_lemmas,
                "lemma_devanagari": devanagari_lemmas,
                "token_count": len(surfaces),
                "features": features,
            }
        )

        current_id = None
        surfaces = []
        lemmas = []
        features = []

    for raw_line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("# id ="):
            flush()
            current_id = line.split("=", 1)[1].strip()
            continue

        if line.startswith("#"):
            continue

        parts = line.split("\t")

        if not parts:
            continue

        surface = parts[0].strip()
        lemma = parts[1].strip() if len(parts) > 1 else surface
        feature = parts[2].strip() if len(parts) > 2 else ""

        if surface:
            surfaces.append(surface)
            lemmas.append(lemma or surface)
            features.append(feature)

    flush()

    return passages


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a local Stage-6D retrieval corpus from the "
            "Ambuda sanitized DCS repository."
        )
    )

    parser.add_argument(
        "--source",
        default="knowledge/dcs-ambuda",
    )
    parser.add_argument(
        "--output",
        default="knowledge/stage6d_dcs",
    )

    args = parser.parse_args()

    try:
        from vidyut import lipi
    except Exception as exc:
        raise RuntimeError(
            "Activate venv-stage6; Vidyut is required for SLP1↔Devanagari."
        ) from exc

    source = Path(args.source)
    output = Path(args.output)

    if not source.is_dir():
        raise FileNotFoundError(
            f"DCS source directory does not exist: {source}"
        )

    text_files = sorted(source.glob("*.txt"))

    if not text_files:
        raise RuntimeError(
            f"No .txt DCS files found in: {source}"
        )

    output.mkdir(parents=True, exist_ok=True)

    passages_path = output / "passages.jsonl"
    metadata_path = output / "metadata.json"

    total_passages = 0
    total_tokens = 0
    file_counts: Dict[str, int] = {}

    with passages_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for text_file in text_files:
            passages = _parse_dcs_file(
                text_file,
                lipi=lipi,
            )

            file_counts[text_file.name] = len(passages)

            for passage in passages:
                handle.write(
                    json.dumps(
                        passage,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_passages += 1
                total_tokens += int(
                    passage["token_count"]
                )

    metadata = {
        "preparation_version": PREP_VERSION,
        "source_kind": "Ambuda sanitized Digital Corpus of Sanskrit data",
        "source_directory": str(source),
        "output_passages": str(passages_path),
        "files_processed": len(text_files),
        "passages": total_passages,
        "tokens": total_tokens,
        "passages_per_file": file_counts,
        "expected_source_license": "CC-BY-4.0 per ambuda-org/dcs README",
        "pipeline_note": (
            "This corpus is retrieval evidence only. Retrieved passages may not "
            "be promoted directly to manuscript transcription."
        ),
    }

    _write_json(
        metadata_path,
        metadata,
    )

    print(
        "Stage 6D corpus preparation completed."
    )
    print(
        "Files processed:",
        len(text_files),
    )
    print(
        "Passages:",
        total_passages,
    )
    print(
        "Tokens:",
        total_tokens,
    )
    print(
        "Passage file:",
        passages_path,
    )
    print(
        "Metadata:",
        metadata_path,
    )


if __name__ == "__main__":
    main()
