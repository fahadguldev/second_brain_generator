"""Unit tests for Second Brain App topic classification and config integration."""

import tempfile
from pathlib import Path
import pytest
import yaml

from src.classifier import (
    classify_topics,
    determine_domain,
    detect_language,
    get_brain_config,
    get_topics_lexicon,
)


def test_classify_topics_technical():
    """Technical terms are matched to correct topics."""
    text = "Building a REST API with FastAPI and Python."
    topics = classify_topics(text=text, filename="project_readme.md")
    assert "python" in topics
    assert determine_domain(topics) == "professional"


def test_classify_topics_personal_family():
    """Family keywords match 'family' topic and 'personal' domain."""
    text = "Spending time with my parents and siblings at home."
    topics = classify_topics(text=text, filename="life_notes.md")
    assert "family" in topics
    assert determine_domain(topics) == "personal"


def test_classify_topics_personal_marriage():
    """Marriage keywords match 'marriage' topic and 'personal' domain."""
    text = "Thoughts on when to settle down and finding a good life partner for marriage."
    topics = classify_topics(text=text, filename="thoughts.md")
    assert "marriage" in topics
    assert determine_domain(topics) == "personal"


def test_classify_topics_personal_studies():
    """Studies keywords match 'studies' topic and 'personal' domain."""
    text = "Preparing for my semester exams and university courses."
    topics = classify_topics(text=text)
    assert "studies" in topics
    assert determine_domain(topics) == "personal"


def test_classify_topics_personal_future():
    """Future vision and goals match 'future' topic."""
    text = "My 5-year vision and roadmap for long-term ambition."
    topics = classify_topics(text=text)
    assert "future" in topics
    assert determine_domain(topics) == "personal"


def test_domain_uses_majority_of_matched_categories():
    """The domain follows the larger category group for a mixed chunk."""
    assert determine_domain(["projects", "python", "future"]) == "professional"
    assert determine_domain(["family", "future", "projects"]) == "personal"
    assert determine_domain(["family", "projects"]) == "professional"


def test_filename_contributes_to_topics():
    """Filename alone triggers category match even if body text is vague."""
    topics = classify_topics(text="Some generic overview text.", filename="aws_cloud_guide.md")
    assert "cloud" in topics


def test_language_detection():
    """Detects english and hinglish correctly."""
    assert detect_language("This is a clean English sentence.") == "english"
    assert detect_language("kya haal hai bhai, kaise ho aap?") == "hinglish"
    assert detect_language("") is None


def test_category_crud_operations():
    """Verify listing, saving, and removing categories dynamically."""
    from src.classifier import list_categories, save_category, remove_category

    # 1. List
    initial = list_categories()
    assert len(initial) > 0
    names = [c["name"] for c in initial]
    assert "python" in names

    # 2. Add new category
    created = save_category(name="fitness_test", keywords=["gym", "workout", "cardio"], domain="personal")
    assert created["name"] == "fitness_test"
    assert "gym" in created["keywords"]
    assert created["domain"] == "personal"

    # Verify classification immediately works with the new category
    test_topics = classify_topics(text="Going to the gym for my daily workout.")
    assert "fitness_test" in test_topics
    assert determine_domain(test_topics) == "personal"

    # 3. Clean up by removing the test category
    deleted = remove_category("fitness_test")
    assert deleted is True

    # Verify it is removed
    after = list_categories()
    assert "fitness_test" not in [c["name"] for c in after]
