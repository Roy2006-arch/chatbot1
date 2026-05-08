"""
multimodal/chat_integration.py
-------------------------------
FastAPI endpoint for single-file multimodal processing.
Accepts image/PDF uploads, routes through MultimodalPipeline,
returns extracted JSON with embedding, injects results into Orchestrator context.

Usage:
    from multimodal.chat_integration import multimodal_router
    app.include_router(multimodal_router, prefix="/api/chat")
"""

import json
import logging
import os
import uuid
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, UploadFile, File, HTTPException

from multimodal.pipeline import MultimodalPipeline
from multimodal.schema import MultimodalExample, MediaType

log = logging.getLogger("chatbot.multimodal_chat")

router = APIRouter(tags=["Multimodal Chat"])

_pipeline: Optional[MultimodalPipeline] = None


def get_pipeline() -> MultimodalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MultimodalPipeline()
    return _pipeline


def process_upload(
    file_bytes: bytes,
    filename: str,
    pipeline: Optional[MultimodalPipeline] = None,
) -> Dict[str, Any]:
    """
    Process a single uploaded file through the MultimodalPipeline.
    Returns a JSON-serializable dict with extracted data.
    """
    pipe = pipeline or get_pipeline()
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return _process_pdf_upload(file_bytes, filename, pipe)
    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
        return _process_image_upload(file_bytes, filename, pipe)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")


def _process_image_upload(
    file_bytes: bytes,
    filename: str,
    pipeline: MultimodalPipeline,
) -> Dict[str, Any]:
    """Process a single image upload through OCR + captioning + embedding."""
    import tempfile
    import shutil

    tmp_dir = tempfile.mkdtemp(prefix="mm_chat_")
    tmp_path = os.path.join(tmp_dir, filename)
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    try:
        mt = MediaType.IMAGE
        if pipeline.preprocessor.enabled:
            mt = pipeline.preprocessor.detect_media_type(tmp_path)

        ocr_result = None
        if pipeline.ocr.enabled:
            ocr_result = pipeline.ocr.extract(tmp_path)

        ocr_text = ocr_result.raw_text if ocr_result else ""
        ocr_conf = ocr_result.confidence if ocr_result else 0.0

        caption = ""
        if pipeline.visual_qa.enabled:
            caption = pipeline.visual_qa.generate_caption(tmp_path, ocr_text)

        embedding = None
        if pipeline.embedder.enabled:
            ex = MultimodalExample(
                prompt=f"Describe this image: {filename}",
                response=caption or ocr_text,
                image_path=tmp_path,
                ocr_text=ocr_text,
                media_type=mt,
            )
            ex = pipeline.embedder.embed_batch([ex])[0]
            embedding = ex.image_embedding

        return {
            "filename": filename,
            "media_type": mt.value,
            "ocr_text": ocr_text,
            "ocr_confidence": ocr_conf,
            "caption": caption,
            "embedding": embedding[:128] if embedding else None,
            "embedding_dim": len(embedding) if embedding else 0,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _process_pdf_upload(
    file_bytes: bytes,
    filename: str,
    pipeline: MultimodalPipeline,
) -> Dict[str, Any]:
    """Process a single PDF upload through text extraction + embedding."""
    import tempfile
    import shutil

    tmp_dir = tempfile.mkdtemp(prefix="mm_chat_")
    tmp_path = os.path.join(tmp_dir, filename)
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    try:
        doc = pipeline.pdf_processor.process(tmp_path)
        if doc is None:
            raise HTTPException(status_code=422, detail="Failed to process PDF")

        text_content = "\n".join(
            r.raw_text for r in doc.ocr_results if r and r.raw_text
        )

        embedding = None
        if pipeline.embedder.enabled and text_content:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            emb = embedder.encode([text_content[:512]])[0]
            embedding = emb.tolist()

        return {
            "filename": filename,
            "media_type": "pdf",
            "page_count": doc.page_count,
            "total_chars": doc.total_chars,
            "text": text_content[:2000],
            "text_truncated": len(text_content) > 2000,
            "num_embedded_images": len(doc.embedded_images),
            "embedding": embedding[:128] if embedding else None,
            "embedding_dim": len(embedding) if embedding else 0,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/upload")
async def chat_upload(file: UploadFile = File(...)):
    """
    Accept an image or PDF upload, process through MultimodalPipeline,
    and return extracted JSON with embedding.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 20MB)")

    try:
        result = process_upload(contents, file.filename)
        return {"status": "success", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        log.error("[MultimodalChat] Upload error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
