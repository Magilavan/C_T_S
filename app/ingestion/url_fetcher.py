"""Secure URL downloader & validator for pharmaceutical PDFs (e.g. RxAbbVie prescribing information).

Supports:
  - Custom browser User-Agent headers to avoid anti-bot blocks
  - SSL/TLS verification
  - Magic-byte (%PDF-) validation
  - Streaming download with file size limits (up to 100MB)
  - Automatic drug name extraction from RxAbbVie URL paths
"""
import logging
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Default browser header to prevent 403 Forbidden from pharmaceutical CDNs
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

MAX_DOWNLOAD_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


class PDFDownloadError(ValueError):
    """Raised when downloading or validating a remote PDF fails."""
    pass


def infer_drug_name_from_url(url: str) -> str:
    """Infer the drug brand name from common URL paths like https://www.rxabbvie.com/pdf/rinvoq_pi.pdf."""
    try:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path
        filename = Path(path).stem.lower()

        # Remove common suffixes like _pi, _medguide, _ifu, _label, _ppi
        clean_stem = re.sub(r"(_pi|_medguide|_ifu|_label|_ppi|_package_insert|_patient_info)$", "", filename)
        # Remove trailing dashes or underscores
        clean_stem = re.sub(r"[-_]+", " ", clean_stem).strip()

        if clean_stem and len(clean_stem) >= 2:
            return clean_stem.upper()
    except Exception as e:
        logger.debug("Failed to infer drug name from URL %s: %s", url, e)
    return "UNKNOWN_DRUG"


def fetch_pdf_from_url(
    url: str,
    target_dir: str | None = None,
    timeout_seconds: int = 30,
) -> tuple[str, str, int]:
    """Download a PDF from a URL, validate its PDF format, and save to disk.

    Args:
        url: Remote URL pointing to the PDF document (e.g. https://www.rxabbvie.com/pdf/rinvoq_pi.pdf)
        target_dir: Optional directory to save the file. If None, uses a secure temp file.
        timeout_seconds: Connection timeout in seconds.

    Returns:
        tuple of (saved_file_path, inferred_drug_name, file_size_bytes)

    Raises:
        PDFDownloadError: If download fails, response is not a PDF, or file exceeds size limit.
    """
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or parsed.scheme.lower() not in ("http", "https"):
        raise PDFDownloadError(f"Invalid URL scheme '{parsed.scheme}'. Must be HTTP or HTTPS.")

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/pdf,application/octet-stream,*/*",
        },
    )

    inferred_drug = infer_drug_name_from_url(url)

    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
        filename = Path(parsed.path).name or f"{inferred_drug.lower()}_pi.pdf"
        dest_path = str(Path(target_dir) / filename)
    else:
        temp_fd, dest_path = tempfile.mkstemp(suffix=".pdf", prefix="rx_download_")
        os.close(temp_fd)

    total_bytes = 0
    first_chunk = True

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            status_code = response.getcode()
            if status_code != 200:
                raise PDFDownloadError(f"HTTP error {status_code} fetching PDF from {url}")

            content_type = response.headers.get("Content-Type", "").lower()
            # If server sends content-type, check that it's not pure HTML/error page
            if "text/html" in content_type:
                raise PDFDownloadError(f"URL returned HTML instead of a PDF document: {url}")

            with open(dest_path, "wb") as f_out:
                while True:
                    chunk = response.read(64 * 1024)  # 64 KB chunk
                    if not chunk:
                        break

                    # Magic bytes check on the first chunk
                    if first_chunk:
                        first_chunk = False
                        if not chunk.startswith(b"%PDF-"):
                            raise PDFDownloadError(
                                f"Downloaded file does not have a valid PDF header (%PDF-): {url}"
                            )

                    total_bytes += len(chunk)
                    if total_bytes > MAX_DOWNLOAD_SIZE_BYTES:
                        raise PDFDownloadError(
                            f"Remote PDF exceeds maximum size of 100 MB (downloaded {total_bytes} bytes)"
                        )

                    f_out.write(chunk)

        if total_bytes == 0:
            raise PDFDownloadError(f"Downloaded PDF is 0 bytes: {url}")

        logger.info("Successfully fetched %d bytes from '%s' -> saved to %s", total_bytes, url, dest_path)
        return dest_path, inferred_drug, total_bytes

    except Exception as exc:
        # Clean up partial download
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        if isinstance(exc, PDFDownloadError):
            raise
        raise PDFDownloadError(f"Failed to fetch PDF from URL '{url}': {exc}") from exc
