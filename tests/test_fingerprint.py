"""Tests for prompt fingerprinting."""

import pytest
from invokelens_sdk.fingerprint import compute_fingerprint, compute_similarity


class TestComputeFingerprint:
    def test_simple_prompt(self):
        fp = compute_fingerprint("Hello, how are you?")
        assert fp["prompt_hash"]  # non-empty SHA-256
        assert fp["structure_hash"]
        assert fp["char_count"] == 19
        assert fp["word_count"] == 4
        assert fp["line_count"] == 1
        assert fp["template_vars"] == []

    def test_multiline_prompt(self):
        prompt = "Line 1\nLine 2\nLine 3"
        fp = compute_fingerprint(prompt)
        assert fp["line_count"] == 3
        assert fp["word_count"] == 6

    def test_template_variable_extraction(self):
        prompt = "Hello {name}, your query is: {query}"
        fp = compute_fingerprint(prompt)
        assert fp["template_vars"] == ["name", "query"]  # sorted

    def test_structure_hash_stability(self):
        """Same template with different variable values -> same structure_hash."""
        fp1 = compute_fingerprint("Hello {name}, welcome to {place}")
        fp2 = compute_fingerprint("Hello {user}, welcome to {location}")
        # Both have {VAR} replacements in same positions -> same structure_hash
        assert fp1["structure_hash"] == fp2["structure_hash"]

    def test_different_prompts_different_hashes(self):
        fp1 = compute_fingerprint("What is the weather?")
        fp2 = compute_fingerprint("Translate this text to French")
        assert fp1["prompt_hash"] != fp2["prompt_hash"]
        assert fp1["structure_hash"] != fp2["structure_hash"]

    def test_empty_prompt(self):
        fp = compute_fingerprint("")
        assert fp["char_count"] == 0
        assert fp["word_count"] == 0
        assert fp["line_count"] == 0
        assert fp["template_vars"] == []

    def test_normalization(self):
        """Leading/trailing whitespace and case should not affect hash."""
        fp1 = compute_fingerprint("Hello World")
        fp2 = compute_fingerprint("  hello world  ")
        assert fp1["prompt_hash"] == fp2["prompt_hash"]

    def test_complex_template_vars(self):
        prompt = "System: {system_prompt}\nUser: {user_input}\nContext: {ctx_data}"
        fp = compute_fingerprint(prompt)
        assert fp["template_vars"] == ["ctx_data", "system_prompt", "user_input"]

    def test_no_false_template_vars_in_json(self):
        """JSON curly braces should not be detected as template vars."""
        prompt = '{"key": "value", "count": 42}'
        fp = compute_fingerprint(prompt)
        # JSON keys like {"key": ...} have a quote after {, so they don't match {var}
        assert fp["template_vars"] == []


class TestComputeSimilarity:
    def test_identical_prompts(self):
        fp1 = compute_fingerprint("Hello World")
        fp2 = compute_fingerprint("Hello World")
        assert compute_similarity(fp1, fp2) == 1.0

    def test_same_template_different_values(self):
        fp1 = compute_fingerprint("Hello {name}")
        fp2 = compute_fingerprint("Hello {user}")
        # Same structure_hash -> 0.9
        assert compute_similarity(fp1, fp2) == 0.9

    def test_completely_different_prompts(self):
        fp1 = compute_fingerprint("a")
        fp2 = compute_fingerprint("This is a very long prompt with many words and sentences that should be completely different from a single character")
        similarity = compute_similarity(fp1, fp2)
        assert similarity < 0.5

    def test_similar_length_prompts(self):
        fp1 = compute_fingerprint("Hello, how are you doing today?")
        fp2 = compute_fingerprint("Greetings, what is your status?")
        similarity = compute_similarity(fp1, fp2)
        # Similar length/structure should yield moderate similarity
        assert 0.3 < similarity < 1.0

    def test_empty_fingerprint(self):
        fp = compute_fingerprint("Hello")
        assert compute_similarity(fp, {}) == 0.0
        assert compute_similarity({}, fp) == 0.0
        assert compute_similarity({}, {}) == 0.0

    def test_none_fingerprint(self):
        fp = compute_fingerprint("Hello")
        assert compute_similarity(fp, None) == 0.0
        assert compute_similarity(None, fp) == 0.0

    def test_similarity_is_bounded(self):
        """Similarity should always be between 0.0 and 1.0."""
        fp1 = compute_fingerprint("Short")
        fp2 = compute_fingerprint("A" * 10000)
        sim = compute_similarity(fp1, fp2)
        assert 0.0 <= sim <= 1.0
