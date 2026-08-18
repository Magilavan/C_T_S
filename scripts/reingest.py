"""Re-ingest rinvoq_pi.pdf with updated section detector and chunker."""
import os
import sys

# Ensure root dir in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.pipeline import ingest_pdf

if __name__ == "__main__":
    pdf_path = "rinvoq_pi.pdf"
    if os.path.exists(pdf_path):
        print(f"Re-ingesting {pdf_path}...")
        res = ingest_pdf(pdf_path, drug_name="Rinvoq")
        print("Done:", res)
    else:
        print(f"File {pdf_path} not found.")
