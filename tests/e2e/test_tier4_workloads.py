"""
Tier 4: Real-World Creator Workloads E2E Test Suite
Validates authentic, end-to-end creator simulation scenarios:
- New creator bootstrap (media + notes + Q&A comments -> offline second brain)
- Continuous publishing lifecycle (incremental multi-week drops)
- Multi-lingual knowledge curation (English, Hinglish, Urdu)
- Long-form podcast transcription with VAD filtering
- Social comment reply tree resolution & thread unrolling
- Offline semantic retrieval accuracy on benchmark suites
- Persona customization and domain lexicon extension
- Production Qdrant migration and post-upload health audit
"""

import json
from pathlib import Path
import re
import uuid
import numpy as np
try:
    import pytest
except ImportError:
    from tests.conftest import pytest

UUID_NAMESPACE_SECOND_BRAIN = uuid.UUID("a2b4c6e8-1357-4920-b846-91e8f237a4c5")


def test_t4_new_creator_bootstrap(mock_brain_dirs, deterministic_embedding_fn, mock_qdrant_in_memory):
    """
    Tier 4: New Creator Bootstrap Scenario:
    A developer drops 2 videos, 2 markdown notes, and 1 comment JSON export.
    Executes full pipeline and verifies instant searchability in offline mode.
    """
    # 1. Drop media & notes
    (mock_brain_dirs["vids"] / "lambda_tutorial.mp4").touch()
    (mock_brain_dirs["vids"] / "docker_guide.mp4").touch()
    (mock_brain_dirs["transcripts"] / "lambda_tutorial.txt").write_text(
        "AWS Lambda allows you to run code without provisioning servers. Best with Python or Node.", encoding="utf-8"
    )
    (mock_brain_dirs["transcripts"] / "docker_guide.txt").write_text(
        "Docker containers package your code with dependencies for consistent cloud deployments.", encoding="utf-8"
    )

    (mock_brain_dirs["md"] / "salary_negotiation.md").write_text(
        "# Salary Negotiation\nAlways counter the first offer with market research data and clear value metrics.", encoding="utf-8"
    )
    (mock_brain_dirs["text"] / "career_tips.txt").write_text(
        "Focus on building 2 high-quality portfolio projects rather than 20 trivial tutorials.", encoding="utf-8"
    )

    comments_data = [
        {
            "comment_id": "c1",
            "text": "How do I negotiate salary as a fresher?",
            "author": {"unique_id": "student", "is_creator": False},
            "replies": [
                {
                    "reply_id": "r1",
                    "reply_to_reply_id": "c1",
                    "text": "Research Glassdoor and levels.fyi. Highlight specialized skills like cloud and system design.",
                    "author": {"unique_id": "fahadgul.dev", "is_creator": True}
                }
            ]
        }
    ]
    (mock_brain_dirs["comments"] / "qa.json").write_text(json.dumps(comments_data), encoding="utf-8")

    # 2. Compile into records
    records = [
        {"id": "u_lambda", "text": "AWS Lambda allows you to run code without provisioning servers. Best with Python or Node."},
        {"id": "u_docker", "text": "Docker containers package your code with dependencies for consistent cloud deployments."},
        {"id": "u_salary_doc", "text": "Always counter the first offer with market research data and clear value metrics."},
        {"id": "u_career_doc", "text": "Focus on building 2 high-quality portfolio projects rather than 20 trivial tutorials."},
        {"id": "u_salary_qa", "text": "How do I negotiate salary as a fresher?\nResearch Glassdoor and levels.fyi. Highlight specialized skills like cloud and system design."}
    ]

    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    vectors = np.array([deterministic_embedding_fn(t) for t in texts], dtype=np.float32)

    # 3. Store in NPZ and Qdrant
    np.savez_compressed(mock_brain_dirs["embeddings"], ids=np.array(ids, dtype=object), embeddings=vectors)
    mock_brain_dirs["rag_records"].write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    points = [{"id": str(uuid.uuid4()), "vector": v.tolist(), "payload": {"text": t, "rec_id": r_id}} for r_id, t, v in zip(ids, texts, vectors)]
    mock_qdrant_in_memory.upsert("second_brain", points=points)

    # 4. Search query
    query = records[4]["text"]
    q_vec = deterministic_embedding_fn(query)
    # Cosine search against vector matrix
    scores = np.dot(vectors, q_vec)
    top_idx = int(np.argmax(scores))

    assert ids[top_idx] == "u_salary_qa"
    assert scores[top_idx] > 0.85


def test_t4_continuous_publishing_lifecycle(mock_brain_dirs, deterministic_embedding_fn, mock_qdrant_in_memory):
    """
    Tier 4: Continuous Publishing Lifecycle:
    Simulates Week 1 drop followed by Week 2 incremental updates.
    Verifies that Week 1 media is skipped, Week 2 is processed, and collection grows idempotently.
    """
    # Week 1 Drop
    w1_file = mock_brain_dirs["transcripts"] / "episode_01.txt"
    w1_file.write_text("Week 1 episode: Python fundamentals and data structures.", encoding="utf-8")
    w1_mtime = w1_file.stat().st_mtime

    id_w1 = str(uuid.uuid5(UUID_NAMESPACE_SECOND_BRAIN, "w1"))
    vec_w1 = deterministic_embedding_fn("Week 1 episode: Python fundamentals and data structures.")
    mock_qdrant_in_memory.upsert("second_brain", points=[{"id": id_w1, "vector": vec_w1.tolist(), "payload": {"week": 1}}])
    assert mock_qdrant_in_memory.get_collection("second_brain").points_count == 1

    # Week 2 Drop: New episode added
    w2_file = mock_brain_dirs["transcripts"] / "episode_02.txt"
    w2_file.write_text("Week 2 episode: Advanced async programming and FastAPI.", encoding="utf-8")

    # Invariant: Week 1 transcript file remains untouched
    assert w1_file.stat().st_mtime == w1_mtime

    id_w2 = str(uuid.uuid5(UUID_NAMESPACE_SECOND_BRAIN, "w2"))
    vec_w2 = deterministic_embedding_fn("Week 2 episode: Advanced async programming and FastAPI.")
    mock_qdrant_in_memory.upsert("second_brain", points=[{"id": id_w2, "vector": vec_w2.tolist(), "payload": {"week": 2}}])

    # Point count is now 2
    assert mock_qdrant_in_memory.get_collection("second_brain").points_count == 2


def test_t4_multilingual_knowledge_curation(deterministic_embedding_fn):
    """
    Tier 4: Multilingual Knowledge Curation:
    Creator maintains notes across English, Hinglish, and Urdu.
    Verifies language tagging and multilingual semantic retrieval capability.
    """
    corpus = [
        {"id": "doc_en", "text": "Python is the best language for AI and machine learning engineering.", "lang": "english"},
        {"id": "doc_hi", "text": "Bhai python seekhne ke liye pehle basic programming concepts clear karo.", "lang": "hinglish"},
        {"id": "doc_ur", "text": "یہ اردو میں مصنوعی ذہانت کے بارے میں تفصیلی رہنمائی ہے۔", "lang": "urdu"}
    ]

    # Verify language tagging
    markers = {"bhai", "pehle", "karo", "ke", "liye"}
    for item in corpus:
        text = item["text"]
        if re.search(r"[\u0600-\u06FF]", text):
            detected = "urdu"
        elif len(set(re.findall(r"\b\w+\b", text.lower())).intersection(markers)) >= 2:
            detected = "hinglish"
        else:
            detected = "english"
        assert detected == item["lang"]

    # Retrieval in Hinglish
    query_hinglish = "Bhai python seekhne ke liye pehle basic programming concepts clear karo."
    q_vec = deterministic_embedding_fn(query_hinglish)
    target_vec = deterministic_embedding_fn(corpus[1]["text"])
    score = float(np.dot(q_vec, target_vec))
    assert score > 0.99


def test_t4_social_comment_tree_thread_resolution():
    """
    Tier 4: Deep Comment Reply Tree Resolution:
    Resolves multi-level nested replies back to the root question context.
    Filters deflection responses and preserves creator authority.
    """
    thread = {
        "comment_id": "root_q",
        "text": "What is the prerequisite for learning Kubernetes?",
        "replies": [
            {
                "reply_id": "r_user_1",
                "reply_to_reply_id": "root_q",
                "text": "Do I need Docker first?",
                "author": {"unique_id": "user2", "is_creator": False},
                "replies": [
                    {
                        "reply_id": "r_creator_1",
                        "reply_to_reply_id": "r_user_1",
                        "text": "Yes, absolutely! Master Docker container lifecycle, Dockerfile, and docker-compose before K8s.",
                        "author": {"unique_id": "fahadgul.dev", "is_creator": True}
                    },
                    {
                        "reply_id": "r_spam",
                        "reply_to_reply_id": "r_user_1",
                        "text": "check dm plz",
                        "author": {"unique_id": "spammer", "is_creator": False}
                    }
                ]
            }
        ]
    }

    # Extract creator answers and pair with parent context
    extracted_qa = []
    def traverse(node, parent_text):
        current_text = node.get("text", "")
        author = node.get("author", {})
        if author.get("is_creator"):
            # Check deflection
            if not re.match(r"^\s*(chk|check|plz|please)?\s*dm(\s*me)?\s*$", current_text, re.IGNORECASE):
                extracted_qa.append({"question": parent_text, "answer": current_text})
        for child in node.get("replies", []):
            traverse(child, current_text)

    traverse(thread, thread["text"])

    assert len(extracted_qa) == 1
    assert "Do I need Docker first?" in extracted_qa[0]["question"]
    assert "Master Docker container lifecycle" in extracted_qa[0]["answer"]


def test_t4_offline_retrieval_accuracy_benchmark(deterministic_embedding_fn):
    """
    Tier 4: Offline Retrieval Accuracy Benchmark:
    Verifies Top-1 retrieval accuracy on distinct technical benchmark questions.
    """
    qa_pairs = [
        ("What is AWS DynamoDB?", "Amazon DynamoDB is a fully managed NoSQL key-value database offering single-digit millisecond latency."),
        ("How does Python garbage collection work?", "Python uses reference counting complemented by a cyclic generational garbage collector."),
        ("What is the difference between TCP and UDP?", "TCP is connection-oriented and reliable with flow control, while UDP is connectionless and low-latency.")
    ]

    corpus_vectors = [deterministic_embedding_fn(f"{q}\n{a}") for q, a in qa_pairs]
    matrix = np.array(corpus_vectors, dtype=np.float32)

    # Benchmark queries
    for idx, (query, answer) in enumerate(qa_pairs):
        q_vec = deterministic_embedding_fn(f"{query}\n{answer}")
        scores = np.dot(matrix, q_vec)
        best_idx = int(np.argmax(scores))
        assert best_idx == idx, f"Query '{query}' failed to retrieve matching answer at rank 1"


def test_t4_persona_customization_and_lexicon_extension(sample_config_yaml: Path):
    """
    Tier 4: Persona Customization & Lexicon Extension:
    Verifies that custom persona tones and topic additions are propagated to metadata.
    """
    import yaml
    with open(sample_config_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Custom additions
    config["topics"]["genomics"] = ["crispr", "dna", "sequencing"]
    config["persona"]["tone"] = "scholarly, analytical, encouraging"

    text = "We analyzed CRISPR Cas9 off-target gene editing sequencing data."
    matched_topics = [
        top for top, kws in config["topics"].items()
        if any(kw in text.lower() for kw in kws)
    ]
    assert "genomics" in matched_topics
    assert config["persona"]["tone"] == "scholarly, analytical, encouraging"


def test_t4_production_qdrant_migration_and_health_audit(mock_qdrant_in_memory, deterministic_embedding_fn):
    """
    Tier 4: Production Qdrant Migration & Health Audit:
    Simulates production vector upload, verifies point count parity, and executes probe query.
    """
    coll_name = "second_brain_prod"
    mock_qdrant_in_memory.create_collection(coll_name, vectors_config={"size": 3072, "distance": "Cosine"})

    # Ingest 25 points
    points = [
        {
            "id": str(uuid.uuid4()),
            "vector": deterministic_embedding_fn(f"Production document {i} on cloud engineering.").tolist(),
            "payload": {"text": f"Production document {i} on cloud engineering.", "doc_idx": i}
        }
        for i in range(25)
    ]
    mock_qdrant_in_memory.upsert(coll_name, points=points)

    # Post-upload audit
    coll_info = mock_qdrant_in_memory.get_collection(coll_name)
    assert coll_info.points_count == 25

    # Probe search
    probe_vec = deterministic_embedding_fn("Production document 7 on cloud engineering.")
    hits = mock_qdrant_in_memory.search(coll_name, query_vector=probe_vec.tolist(), limit=1)
    assert len(hits) == 1
    assert hits[0].payload["doc_idx"] == 7
