from __future__ import annotations

import argparse
import json
import platform
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "krutrim-ai-labs/Vyakyarth"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 6D.2 embedding compatibility test for Sanskrit semantic retrieval. "
            "This is diagnostic only and does not modify manuscript artifacts."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    args = parser.parse_args()

    print("=== ENVIRONMENT ===")
    print("machine:", platform.machine())
    print("python:", platform.python_version())
    print("torch:", torch.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("selected device:", device)
    print()

    print("=== MODEL ===")
    print("loading:", args.model)

    model = SentenceTransformer(
        args.model,
        device=device,
    )

    print("model loaded: OK")
    print("embedding dimension:", model.get_sentence_embedding_dimension())
    print("max sequence length:", model.max_seq_length)
    print()

    # Diagnostic strings only.
    #
    # The first two strings are noisy HTR-like evidence observed in the current
    # manuscript experiment. The comparison documents are used only to check
    # whether a Sanskrit-aware embedding model is sensitive to meaningful
    # neighbourhoods. This test does NOT inject any reference string into the
    # Stage-6 production pipeline.
    queries: List[str] = [
        "श्रास्वरस्वत्य् एनमः",
        "अग्नीगाणज्राग्",
        "द्रास् ब्राण जार्यम्",
    ]

    documents: List[str] = [
        "श्रीसरस्वत्यै नमः",
        "श्रीगणेशाय नमः",
        "अग्नये नमः",
        "विद्या ददाति विनयम्",
        "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः",
        "सर्वे भवन्तु सुखिनः",
    ]

    print("=== ENCODING ===")

    query_embeddings = model.encode(
        queries,
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    document_embeddings = model.encode(
        documents,
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    print("query shape:", query_embeddings.shape)
    print("document shape:", document_embeddings.shape)
    print()

    similarities = (
        query_embeddings
        @ document_embeddings.T
    )

    print("=== SIMILARITY RANKINGS ===")

    output = []

    for q_index, query in enumerate(queries):
        ranking = np.argsort(
            -similarities[q_index]
        )

        rows = []

        print()
        print("QUERY:", query)

        for rank, doc_index in enumerate(
            ranking,
            start=1,
        ):
            score = float(
                similarities[
                    q_index,
                    doc_index,
                ]
            )

            row = {
                "rank": rank,
                "score": round(score, 4),
                "document": documents[
                    int(doc_index)
                ],
            }

            rows.append(row)

            print(
                f"  {rank}. {score:.4f}  "
                f"{documents[int(doc_index)]}"
            )

        output.append(
            {
                "query": query,
                "ranking": rows,
            }
        )

    print()
    print("=== JSON SUMMARY ===")
    print(
        json.dumps(
            {
                "model": args.model,
                "device": device,
                "embedding_dimension": (
                    model.get_sentence_embedding_dimension()
                ),
                "results": output,
                "diagnostic_only": True,
                "production_artifacts_modified": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
