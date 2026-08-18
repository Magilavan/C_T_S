"""Tests for Medical Domain Content Restriction, Empathetic Soft Refusal,
Cross-Channel PDF Ingestion Filtering, and Privacy-Preserving Logging.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.rag.safety_classifier import (
    classify_question,
    ClassificationResult,
    EMPATHETIC_OUT_OF_SCOPE_REFUSAL,
)
from app.rag.chain import handle_chat_message
from app.core.privacy_logger import (
    compute_privacy_hash,
    mask_session_id,
    log_out_of_scope_query,
    log_pdf_rejection,
    _HASH_SALT,
)
from app.ingestion.domain_filter import (
    validate_pdf_domain,
    PDFDomainValidationError,
)
from app.ingestion.pdf_extractor import PageContent


# ── 1. Medical vs Non-Medical Domain Restriction Tests ───────────────────────

class TestMedicalDomainClassification:
    def test_deterministic_out_of_scope_queries(self):
        non_medical_queries = [
            "write a python script to sort a list",
            "debug this code for my react app",
            "invest in bitcoin and crypto trading",
            "who won the super bowl last year?",
            "give me a recipe for chocolate cake",
            "what is the weather in Paris?",
            "solve the math equation 2x + 5 = 15",
            "who is the president of France?",
            "give me relationship and dating advice",
            "tell me a joke about programming",
        ]
        for query in non_medical_queries:
            result = classify_question(query)
            assert result.category == "out_of_scope", f"Expected out_of_scope for: '{query}', got: {result.category}"
            assert result.requires_rag is False
            assert result.risk_level == "low"

    def test_in_scope_medical_queries(self):
        medical_queries = [
            "What is the recommended dosage of RINVOQ for rheumatoid arthritis?",
            "What are the contraindications and boxed warnings for adalimumab?",
            "What adverse reactions are most commonly reported with methotrexate?",
            "How is the drug administered for pediatric patients?",
            "What drug interactions should be monitored with CYP3A4 inhibitors?",
        ]
        for query in medical_queries:
            result = classify_question(query)
            assert result.category in ("general_label", "patient_specific"), (
                f"Expected medical category for: '{query}', got: {result.category}"
            )
            assert result.requires_rag is True

    def test_patient_specific_medical_queries(self):
        patient_queries = [
            "I am taking 15mg of Rinvoq and developed chest pain, what should I do?",
            "My doctor prescribed this medicine but I missed a dose yesterday.",
            "Can I take this medication while pregnant?",
        ]
        for query in patient_queries:
            result = classify_question(query)
            assert result.category in ("patient_specific", "high_risk"), (
                f"Expected patient_specific or high_risk for: '{query}', got: {result.category}"
            )


# ── 2. Empathetic Soft-Tone Refusal Tests ────────────────────────────────────

class TestEmpatheticSoftRefusal:
    def test_out_of_scope_chat_returns_empathetic_refusal_without_rag(self):
        session_id = "test-scope-session-1"
        message = "write a python function to scrape a website"

        with patch("app.rag.chain.resolve_context") as mock_resolve, \
             patch("app.rag.chain.hybrid_retrieve") as mock_retrieve, \
             patch("app.rag.chain.generate_answer") as mock_gen:

            res = handle_chat_message(session_id=session_id, message=message)

            assert res["confidence"] == "out_of_scope"
            assert res["question_category"] == "out_of_scope"
            assert res["citations"] == []
            assert res["answer"] == EMPATHETIC_OUT_OF_SCOPE_REFUSAL
            assert "pharmaceutical" in res["answer"].lower()
            assert "prescribing" in res["answer"].lower()

            # Ensure zero RAG retrieval / generation LLM calls were made
            mock_resolve.assert_not_called()
            mock_retrieve.assert_not_called()
            mock_gen.assert_not_called()


# ── 3. Cross-Channel PDF Ingestion Domain Filtering Tests ───────────────────

class TestPDFIngestionDomainFilter:
    def test_valid_medical_pdf_passes(self):
        valid_page_1 = PageContent(
            page_number=1,
            text="""
            HIGHLIGHTS OF PRESCRIBING INFORMATION
            RINVOQ (upadacitinib) extended-release tablets, for oral use
            Initial U.S. Approval: 2019
            WARNING: SERIOUS INFECTIONS, MORTALITY, MALIGNANCY, MAJOR ADVERSE CARDIOVASCULAR EVENTS
            INDICATIONS AND USAGE: RINVOQ is a Janus kinase (JAK) inhibitor indicated for rheumatoid arthritis.
            DOSAGE AND ADMINISTRATION: The recommended dosage is 15 mg once daily.
            DOSAGE FORMS AND STRENGTHS: Extended-release tablets: 15 mg, 30 mg, 45 mg.
            CONTRAINDICATIONS: Known hypersensitivity to upadacitinib.
            """,
            tables=[],
            images=[],
        )
        valid_page_2 = PageContent(
            page_number=2,
            text="""
            ADVERSE REACTIONS: Most common adverse reactions include upper respiratory tract infections.
            DRUG INTERACTIONS: Strong CYP3A4 inhibitors increase exposure.
            CLINICAL PHARMACOLOGY: Upadacitinib is a selective and reversible JAK inhibitor.
            """,
            tables=[],
            images=[],
        )

        result = validate_pdf_domain([valid_page_1, valid_page_2], drug_name="RINVOQ")
        assert result["valid"] is True
        assert result["section_matches"] >= 3

    def test_non_medical_pdf_rejected(self):
        cookbook_page = PageContent(
            page_number=1,
            text="""
            Grandma's Best Italian Pasta Recipes
            Chapter 1: Homemade Marinara Sauce
            Ingredients: 4 ripe tomatoes, 2 cloves of garlic, 1 tablespoon olive oil, fresh basil.
            Instructions: Heat olive oil in a skillet over medium heat. Sauté minced garlic until fragrant.
            Simmer crushed tomatoes for 30 minutes. Serve over al dente spaghetti pasta.
            """,
            tables=[],
            images=[],
        )

        with pytest.raises(PDFDomainValidationError) as exc_info:
            validate_pdf_domain([cookbook_page], drug_name="PASTA")

        assert "does not appear to be a medical, pharmaceutical" in str(exc_info.value)

    def test_code_tutorial_pdf_rejected(self):
        code_page = PageContent(
            page_number=1,
            text="""
            Introduction to Python and Data Structures
            Chapter 3: Binary Search Trees and Big-O Complexity
            In computer science, a binary search tree is a rooted binary tree data structure
            with the key of each internal node being greater than all keys in the left subtree.
            Run git clone and pip install requirements to get started with the exercise.
            """,
            tables=[],
            images=[],
        )

        with pytest.raises(PDFDomainValidationError) as exc_info:
            validate_pdf_domain([code_page], drug_name="PYTHON")

        assert "does not appear to be a medical" in str(exc_info.value)


# ── 4. Privacy-Preserving Logging Tests ──────────────────────────────────────

class TestPrivacyPreservingLogging:
    def test_hashing_guarantees(self):
        sample_query = "What is the secret medical condition of John Doe?"
        h1 = compute_privacy_hash(sample_query)
        h2 = compute_privacy_hash(sample_query)
        # Consistent deterministic hashing for metrics aggregation
        assert h1 == h2
        # Salted hash prevents raw string recovery
        assert sample_query not in h1
        assert len(h1) == 16

    def test_masked_session_id(self):
        sess = "user-12345-secret-session"
        masked = mask_session_id(sess)
        assert masked.startswith("sess_")
        assert "12345" not in masked

    def test_out_of_scope_metric_event_structure(self):
        raw_msg = "Can you write some javascript code for me?"
        event = log_out_of_scope_query(
            session_id="session-xyz",
            raw_message=raw_msg,
            classification_source="regex",
            reason="non_medical_coding",
        )

        # Critical privacy rule: raw query text must NEVER be in the telemetry event
        assert "Can you write" not in str(event)
        assert "javascript" not in str(event)
        assert event["event_type"] == "out_of_scope_query"
        assert event["query_length_chars"] == len(raw_msg)
        assert event["classification_source"] == "regex"
        assert "timestamp" in event
        assert "query_hash" in event

    def test_pdf_rejection_metric_event_structure(self):
        event = log_pdf_rejection(
            file_name="financial_report_q3.pdf",
            drug_name="FINANCE",
            page_count=12,
            rejection_reason="Not a medical document",
        )

        assert event["event_type"] == "pdf_domain_rejected"
        assert event["page_count"] == 12
        # Filename and drug name are hashed
        assert "financial_report_q3.pdf" not in event["file_name_hash"]
        assert "FINANCE" not in event["drug_name_hash"]
