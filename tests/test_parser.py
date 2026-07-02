from unittest.mock import MagicMock, patch
import pytest
from parsers.parser import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    extract_text,
    clean_text,
    extract_candidate_name,
    extract_contact_info,
    redact_pii,
)

def test_extract_text_from_pdf():
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Hello Pdf World"
    
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]
    
    with patch("PyPDF2.PdfReader", return_value=mock_reader):
        res = extract_text_from_pdf(b"dummy pdf bytes")
        assert res == "Hello Pdf World"

def test_extract_text_from_docx():
    mock_paragraph = MagicMock()
    mock_paragraph.text = "Hello Docx World"
    
    mock_cell = MagicMock()
    mock_cell.text = "Table Cell Content"
    
    mock_row = MagicMock()
    mock_row.cells = [mock_cell]
    
    mock_table = MagicMock()
    mock_table.rows = [mock_row]
    
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_paragraph]
    mock_doc.tables = [mock_table]
    
    with patch("docx.Document", return_value=mock_doc):
        res = extract_text_from_docx(b"dummy docx bytes")
        assert "Hello Docx World" in res
        assert "Table Cell Content" in res

def test_extract_text_from_txt():
    res = extract_text_from_txt(b"Hello Txt World")
    assert res == "Hello Txt World"

def test_extract_text_dispatch():
    # pdf
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Hello Pdf"
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]
    with patch("PyPDF2.PdfReader", return_value=mock_reader):
        assert extract_text("resume.pdf", b"pdfbytes") == "Hello Pdf"

    # docx
    mock_doc = MagicMock()
    mock_doc.paragraphs = []
    mock_doc.tables = []
    with patch("docx.Document", return_value=mock_doc):
        assert extract_text("resume.docx", b"docxbytes") == ""

    # txt
    assert extract_text("resume.txt", b"txtbytes") == "txtbytes"

    # empty
    assert extract_text("resume.txt", b"") == ""

    # unsupported extension
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text("resume.png", b"somebytes")

    # parsing error raises ValueError
    with patch("PyPDF2.PdfReader", side_effect=Exception("Read failure")):
        with pytest.raises(ValueError, match="Failed to parse resume.pdf"):
            extract_text("resume.pdf", b"corruptpdf")

def test_clean_text():
    assert clean_text("") == ""
    # Test bullets conversion
    raw = "• Python\n◦ Java\n▪ C++\n➤ Go"
    cleaned = clean_text(raw)
    assert "- Python" in cleaned
    assert "- Java" in cleaned
    assert "- C++" in cleaned
    assert "- Go" in cleaned
    
    # Test empty lines collapsing and spaces collapsing
    raw_noisy = "Line 1\n\n\nLine 2     With   Spaces"
    assert clean_text(raw_noisy) == "Line 1\nLine 2 With Spaces"

def test_extract_candidate_name():
    # Standard name extraction
    resume_text = "John Doe\nSoftware Engineer\njohn.doe@example.com\n123-456-7890"
    assert extract_candidate_name(resume_text, "Fallback") == "John Doe"

    # Reject lines containing email or phone numbers
    resume_text_no_name = "john.doe@example.com\n123-456-7890\nthisiswaytoolongtobeanameandshouldnotbematchedbytheheuristicatallline"
    assert extract_candidate_name(resume_text_no_name, "Fallback") == "Fallback"

def test_redact_pii():
    text = "John Doe\nSoftware Engineer\njohn.doe@example.com\n123-456-7890\nWorking at Acme"
    redacted = redact_pii(text)
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "John Doe" not in redacted
    assert "John" not in redacted
    assert "Doe" not in redacted

def test_extract_contact_info():
    text = "John Doe\njohn.doe@example.com\n+91 98765 43210\nWorking at Acme"
    contact = extract_contact_info(text)
    assert contact["email"] == "john.doe@example.com"
    assert contact["phone"] == "+91 98765 43210"

    text_none = "John Doe\nWorking at Acme"
    contact_none = extract_contact_info(text_none)
    assert contact_none["email"] == "Not Provided"
    assert contact_none["phone"] == "Not Provided"

