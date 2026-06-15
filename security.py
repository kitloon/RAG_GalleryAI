import hmac
import ipaddress
import os
from urllib.parse import urlparse


def is_admin_authorized(provided_key: str | None, configured_key: str | None) -> bool:
    """Admin auth is disabled unless ADMIN_API_KEY is configured."""
    if not configured_key:
        return True
    if not provided_key:
        return False
    return hmac.compare_digest(provided_key, configured_key)


def safe_upload_filename(filename: str) -> str:
    safe_name = os.path.basename(filename or "").strip()
    if not safe_name:
        raise ValueError("Invalid filename")
    return safe_name


def validate_public_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are supported")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")

    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "localhost.localdomain"}:
        raise ValueError("Localhost URLs are not allowed")

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return parsed.geturl()

    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        raise ValueError("Private or local network URLs are not allowed")
    return parsed.geturl()
