import os
from pathlib import Path

from fastapi import UploadFile

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", Path(__file__).parent.parent / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def save_uploaded_file(file: UploadFile) -> Path:
    """
    Save an uploaded file to the upload directory.

    Returns:
        Path to the saved file.
    """
    dest = UPLOAD_DIR / file.filename
    # Stream to disk to avoid large memory spikes
    with dest.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return dest
