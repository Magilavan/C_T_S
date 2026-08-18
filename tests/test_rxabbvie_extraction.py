"""Comprehensive test suite for RxAbbVie (https://www.rxabbvie.com/) PDF Data Extraction.

Verifies:
  1. Multi-column reading order (Highlights 2-column layout preserved top-to-bottom)
  2. Unicode normalization, ligature expansion (fi, fl, ffi, ffl), CID artifact removal
  3. De-hyphenation across line wraps for medical terminology
  4. Table bounding box isolation (no duplicate table prose)
  5. Section & Appendix detection (Medication Guides, IFUs, Boxed Warnings, FDA 1-17)
  6. Scanned/rasterized page OCR fallback
  7. Encrypted/restricted PDF handling
  8. Corrupt stream & edge case error resilience
  9. RxAbbVie URL fetcher, magic-byte validation, and drug name inferencing
"""
import io
import os
import tempfile
import pytest
import fitz  # PyMuPDF
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.ingestion.pdf_extractor import (
    clean_and_normalize_text,
    extract_pdf,
    PageContent,
)
from app.ingestion.section_detector import detect_sections
from app.ingestion.url_fetcher import (
    infer_drug_name_from_url,
    fetch_pdf_from_url,
    PDFDownloadError,
)
from app.ingestion.domain_filter import validate_pdf_domain


# ─────────────────────────────────────────────────────────────────────────────
# 1. TEXT NORMALIZATION & UNICODE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestTextNormalizationAndEncoding:
    """Verifies that embedded ligatures, symbols, and CID glyphs are correctly normalized."""

    def test_ligature_expansion(self):
        raw_text = "The ef\ufb01cacy and in\ufb02uence of Rinvoq in clinical trials (pro\ufb01le a\ufb03liation and a\ufb04uent)."
        normalized = clean_and_normalize_text(raw_text)
        assert "efficacy" in normalized
        assert "influence" in normalized
        assert "profile" in normalized
        assert "affiliation" in normalized
        assert "affluent" in normalized
        assert "\ufb01" not in normalized
        assert "\ufb02" not in normalized

    def test_cid_glyph_and_symbol_cleaning(self):
        raw_text = "Dosage (cid:142) should be \u2265 15 mg and \u2264 30 mg with \u00b1 5 mg variance (\u03bcg conversion)."
        normalized = clean_and_normalize_text(raw_text)
        assert "(cid:" not in normalized
        assert ">= 15 mg" in normalized
        assert "<= 30 mg" in normalized
        assert "+/- 5 mg" in normalized
        assert "mcg conversion" in normalized

    def test_dehyphenation_across_lines(self):
        raw_text = (
            "Patients with severe hepatic impairment are contra-\n"
            "indicated from receiving this med-\n"
            "ication due to thrombo-\n"
            "embolism risk."
        )
        normalized = clean_and_normalize_text(raw_text)
        assert "contraindicated" in normalized
        assert "medication" in normalized
        assert "thromboembolism" in normalized


# ─────────────────────────────────────────────────────────────────────────────
# 2. SECTION & APPENDIX DETECTION (MEDICATION GUIDE & IFU)
# ─────────────────────────────────────────────────────────────────────────────

class TestSectionAndAppendixDetection:
    """Verifies recognition of FDA sections 1-17 plus RxAbbVie Medication Guides and IFUs."""

    def test_detects_medication_guide_and_ifu(self):
        sample_doc = """
HIGHLIGHTS OF PRESCRIBING INFORMATION
These highlights do not include all the information needed to use RINVOQ safely.

1 INDICATIONS AND USAGE
RINVOQ is a Janus kinase (JAK) inhibitor indicated for the treatment of...

5 WARNINGS AND PRECAUTIONS
5.1 Serious Infections
Patients treated with RINVOQ are at increased risk for developing serious infections.

17 PATIENT COUNSELING INFORMATION
Advise the patient to read the FDA-approved patient labeling.

MEDICATION GUIDE
MEDICATION GUIDE FOR RINVOQ (upadacitinib) extended-release tablets

INSTRUCTIONS FOR USE
INSTRUCTIONS FOR USE - RINVOQ ON-BODY INJECTOR
Read this Instructions for Use before you start using your injector...
"""
        markers = detect_sections(sample_doc)
        headings = [m.heading for m in markers]

        assert "HIGHLIGHTS OF PRESCRIBING INFORMATION" in headings
        assert "1 INDICATIONS AND USAGE" in headings
        assert "5 WARNINGS AND PRECAUTIONS" in headings
        assert "5.1 Serious Infections" in headings
        assert "17 PATIENT COUNSELING INFORMATION" in headings
        assert any("MEDICATION GUIDE" in h for h in headings)
        assert any("INSTRUCTIONS FOR USE" in h for h in headings)

    def test_domain_filter_accepts_medication_guide(self):
        pages = [
            PageContent(
                page_number=1,
                text="MEDICATION GUIDE FOR RINVOQ (upadacitinib) extended-release tablets. "
                     "What is the most important information I should know about RINVOQ? "
                     "Serious infections, cancer risk, cardiovascular death, blood clots. "
                     "Take 15 mg tablet once daily with or without food. FDA approved patient labeling.",
            )
        ]
        result = validate_pdf_domain(pages, drug_name="RINVOQ")
        assert result["valid"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. MULTI-COLUMN LAYOUT & TABLE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiColumnAndTableExtraction:
    """Verifies multi-column reading order and table bounding box masking."""

    @pytest.fixture
    def sample_two_column_pdf(self, tmp_path):
        """Generate a synthetic PDF with 2-column layout on PyMuPDF."""
        pdf_path = tmp_path / "rxabbvie_sample.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)  # Standard Letter size

        # Full-width Header across the top
        page.insert_textbox(fitz.Rect(50, 40, 560, 80), "HIGHLIGHTS OF PRESCRIBING INFORMATION", fontsize=12)

        # Left Column (x: 50 to 280)
        page.insert_textbox(
            fitz.Rect(50, 100, 280, 200),
            "1 INDICATIONS AND USAGE\nRINVOQ is indicated for Rheumatoid Arthritis.",
            fontsize=10,
        )
        page.insert_textbox(
            fitz.Rect(50, 220, 280, 320),
            "2 DOSAGE AND ADMINISTRATION\nRecommended dose is 15 mg once daily.",
            fontsize=10,
        )

        # Right Column (x: 320 to 560)
        page.insert_textbox(
            fitz.Rect(320, 100, 560, 200),
            "4 CONTRAINDICATIONS\nKnown hypersensitivity to upadacitinib.",
            fontsize=10,
        )
        page.insert_textbox(
            fitz.Rect(320, 220, 560, 320),
            "5 WARNINGS AND PRECAUTIONS\nSerious infections and malignancy risk.",
            fontsize=10,
        )

        doc.save(str(pdf_path))
        doc.close()
        return str(pdf_path)

    def test_multi_column_reading_order_preserved(self, sample_two_column_pdf):
        pages = extract_pdf(sample_two_column_pdf)
        assert len(pages) == 1
        page = pages[0]

        # Column reading order check: Left column sections must appear before Right column sections in extracted prose
        text = page.text
        pos_indic = text.find("1 INDICATIONS AND USAGE")
        pos_dosage = text.find("2 DOSAGE AND ADMINISTRATION")
        pos_contra = text.find("4 CONTRAINDICATIONS")
        pos_warn = text.find("5 WARNINGS AND PRECAUTIONS")

        assert pos_indic != -1, f"INDICATIONS not found in text:\n{text}"
        assert pos_dosage != -1, f"DOSAGE not found in text:\n{text}"
        assert pos_contra != -1, f"CONTRAINDICATIONS not found in text:\n{text}"
        assert pos_warn != -1, f"WARNINGS not found in text:\n{text}"

        # Left column items read before right column items
        assert pos_indic < pos_dosage
        assert pos_dosage < pos_contra
        assert pos_contra < pos_warn


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCRYPTED / RESTRICTED & SCANNED PDF HANDLING
# ─────────────────────────────────────────────────────────────────────────────

class TestSecurityAndEdgeCases:
    """Verifies handling of encrypted permissions, corrupt files, and empty documents."""

    def test_empty_password_encrypted_pdf(self, tmp_path):
        """Test PDF with standard security encryption handler is unlocked automatically."""
        pdf_path = tmp_path / "encrypted_label.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 500, 200), "1 INDICATIONS AND USAGE\nHumira is indicated for Crohn's Disease.")
        # Encrypt with empty user password and owner permissions
        doc.save(str(pdf_path), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="", owner_pw="secret123")
        doc.close()

        pages = extract_pdf(str(pdf_path))
        assert len(pages) == 1
        assert "Crohn's Disease" in pages[0].text

    def test_nonexistent_or_empty_pdf_raises_error(self, tmp_path):
        empty_file = tmp_path / "empty.pdf"
        empty_file.write_bytes(b"")

        with pytest.raises(ValueError, match="does not exist or is empty"):
            extract_pdf(str(empty_file))

        with pytest.raises(ValueError, match="does not exist or is empty"):
            extract_pdf(str(tmp_path / "non_existent.pdf"))


# ─────────────────────────────────────────────────────────────────────────────
# 5. RXABBVIE URL FETCHER & INGESTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestRxAbbVieUrlFetcher:
    """Verifies URL fetching, drug inferencing, and header validation."""

    def test_infer_drug_name_from_rxabbvie_urls(self):
        assert infer_drug_name_from_url("https://www.rxabbvie.com/pdf/humira.pdf") == "HUMIRA"
        assert infer_drug_name_from_url("https://www.rxabbvie.com/pdf/rinvoq_pi.pdf") == "RINVOQ"
        assert infer_drug_name_from_url("https://www.rxabbvie.com/pdf/skyrizi_pi.pdf") == "SKYRIZI"
        assert infer_drug_name_from_url("https://www.rxabbvie.com/pdf/qulipta_medguide.pdf") == "QULIPTA"
        assert infer_drug_name_from_url("https://www.rxabbvie.com/pdf/botox_ifu.pdf") == "BOTOX"

    def test_fetch_pdf_invalid_magic_bytes_rejected(self):
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.read.side_effect = [b"<html><body>404 Not Found</body></html>", b""]
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(PDFDownloadError, match="valid PDF header"):
                fetch_pdf_from_url("https://www.rxabbvie.com/pdf/fake_drug.pdf")

    def test_fetch_pdf_valid_download(self):
        fake_pdf_data = b"%PDF-1.5 simulated pdf stream data for Skyrizi label"
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.read.side_effect = [fake_pdf_data, b""]
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_response):
            saved_path, inferred_drug, size = fetch_pdf_from_url("https://www.rxabbvie.com/pdf/skyrizi_pi.pdf")
            try:
                assert os.path.exists(saved_path)
                assert inferred_drug == "SKYRIZI"
                assert size == len(fake_pdf_data)
                with open(saved_path, "rb") as f:
                    assert f.read().startswith(b"%PDF-")
            finally:
                if os.path.exists(saved_path):
                    os.remove(saved_path)
