import os
import tempfile
from typing import Generator, Tuple
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from PIL import Image


def _make_embedding():
    v = np.random.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def mock_sentence_transformers():
    mock_instance = MagicMock()

    def encode_side_effect(sentences, **kwargs):
        if isinstance(sentences, str):
            return _make_embedding()
        return np.stack([_make_embedding() for _ in sentences])

    mock_instance.encode.side_effect = encode_side_effect
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_instance):
        yield


@pytest.fixture
def sample_image_path() -> Generator[str, None, None]:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img = Image.new("RGB", (200, 200), color="red")
    img.save(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def small_image_path() -> Generator[str, None, None]:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img = Image.new("RGB", (16, 16), color="blue")
    img.save(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def landscape_image_path() -> Generator[str, None, None]:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img = Image.new("RGB", (800, 400), color="green")
    img.save(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def large_image_path() -> Generator[str, None, None]:
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img = Image.new("RGB", (2048, 2048), color="blue")
    img.save(path, quality=95)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def screenshot_image_path() -> Generator[str, None, None]:
    fd, path = tempfile.mkstemp(suffix="_screenshot.png")
    os.close(fd)
    img = Image.new("RGB", (400, 800), color="white")
    img.save(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_pdf_path() -> Generator[str, None, None]:
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    minimal_pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n115\n%%EOF"
    with open(path, "wb") as f:
        f.write(minimal_pdf)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def input_dir_with_images(sample_image_path, screenshot_image_path, landscape_image_path) -> Generator[str, None, None]:
    tmpdir = tempfile.mkdtemp()
    import shutil
    for src in [sample_image_path, screenshot_image_path, landscape_image_path]:
        dst = os.path.join(tmpdir, os.path.basename(src))
        shutil.copy2(src, dst)
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)
