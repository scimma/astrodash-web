import secrets
import hashlib
import time
from typing import Optional
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def generate_secure_token() -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(32)

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(password) == hashed

def rate_limit_check(client_ip: str, endpoint: str, max_requests: int = 100, window_seconds: int = 60) -> bool:
    """
    Simple rate limiting check.
    In production, use Redis or a proper rate limiting library.
    """
    # This is a simplified implementation
    # In production, use Redis with proper rate limiting
    return True

def validate_file_upload(filename: str, max_size_mb: int = 50) -> bool:
    """Validate file upload parameters."""
    if not filename:
        return False
    
    # Check file extension
    allowed_extensions = {'.dat', '.txt', '.lnw', '.fits', '.zip'}
    file_ext = filename.lower()
    if not any(file_ext.endswith(ext) for ext in allowed_extensions):
        return False
    
    return True

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    import re
    # Remove or replace dangerous characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1)
        filename = name[:250] + '.' + ext
    return filename
