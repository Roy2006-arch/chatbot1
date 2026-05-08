import json
import os
import tempfile
import pytest
from multimodal.exporters import MultimodalExporter
from multimodal.schema import MultimodalExample, MediaType


def make_example(image_path="", idx=0):
    return MultimodalExample(
        id=f"ex-{idx}",
        prompt=f"Describe image {idx}",
        response=f"This is image {idx}",
        image_path=image_path or f"/path/to/img{idx}.png",
        ocr_text=f"ocr text {idx}",
        caption=f"Caption {idx}",
        media_type=MediaType.IMAGE,
        category="test",
        quality_score=0.8,
    )


class TestMultimodalExporterInit:
    def test_default_config(self):
        e = MultimodalExporter()
        assert e.enabled is True
        assert "llava" in e.formats
        assert "openai_vision" in e.formats
        assert e.train_split == 0.8
        assert e.val_split == 0.1
        assert e.include_images is True

    def test_custom_config(self):
        e = MultimodalExporter(config={
            "enabled": False, "formats": ["jsonl"], "train_split": 0.7,
        })
        assert e.enabled is False
        assert e.formats == ["jsonl"]
        assert e.train_split == 0.7


class TestExportLlava:
    def test_exports_json(self):
        e = MultimodalExporter()
        examples = [make_example(idx=i) for i in range(3)]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            result = e.export_llava(examples, out_path)
            assert os.path.exists(result)
            with open(result) as f:
                data = json.load(f)
            assert len(data) == 3
            assert "image" in data[0]
            assert "conversations" in data[0]
            assert data[0]["conversations"][0]["from"] == "human"
            assert data[0]["conversations"][1]["from"] == "gpt"
        finally:
            os.unlink(out_path)

    def test_skips_examples_without_image(self):
        e = MultimodalExporter()
        examples = [
            MultimodalExample(prompt="Q", response="A"),
            make_example(idx=1),
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            result = e.export_llava(examples, out_path)
            with open(result) as f:
                data = json.load(f)
            assert len(data) == 1  # only the one with image_path
        finally:
            os.unlink(out_path)

    def test_includes_ocr_and_caption(self):
        e = MultimodalExporter()
        ex = make_example(idx=0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            result = e.export_llava([ex], out_path)
            with open(result) as f:
                data = json.load(f)
            assert data[0]["ocr_text"] == "ocr text 0"
            assert data[0]["caption"] == "Caption 0"
        finally:
            os.unlink(out_path)


class TestExportOpenAIVision:
    def test_exports_jsonl(self):
        e = MultimodalExporter()
        examples = [make_example(idx=i) for i in range(2)]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out_path = f.name
        try:
            result = e.export_openai_vision(examples, out_path)
            assert os.path.exists(result)
            with open(result) as f:
                lines = f.readlines()
            assert len(lines) == 2
            record = json.loads(lines[0])
            assert "messages" in record
            assert record["messages"][0]["role"] == "user"
            assert record["messages"][1]["role"] == "assistant"
        finally:
            os.unlink(out_path)

    def test_includes_content_array(self):
        e = MultimodalExporter(config={"include_images": False})
        ex = make_example(idx=0)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out_path = f.name
        try:
            result = e.export_openai_vision([ex], out_path)
            with open(result) as f:
                record = json.loads(f.readline())
            content = record["messages"][0]["content"]
            assert len(content) == 2
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image_url"
        finally:
            os.unlink(out_path)


class TestExportBlip:
    def test_exports_json(self):
        e = MultimodalExporter()
        examples = [make_example(idx=i) for i in range(2)]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            result = e.export_blip(examples, out_path)
            assert os.path.exists(result)
            with open(result) as f:
                data = json.load(f)
            assert len(data) == 2
            assert "image" in data[0]
            assert "caption" in data[0]
        finally:
            os.unlink(out_path)

    def test_includes_qa_pairs(self):
        e = MultimodalExporter()
        ex = make_example(idx=0)
        ex.metadata["qa_pairs"] = [{"question": "Q?", "answer": "A."}]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            result = e.export_blip([ex], out_path)
            with open(result) as f:
                data = json.load(f)
            assert "qa_pairs" in data[0]
        finally:
            os.unlink(out_path)


class TestExportJsonl:
    def test_exports_jsonl(self):
        e = MultimodalExporter()
        examples = [make_example(idx=i) for i in range(2)]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out_path = f.name
        try:
            result = e.export_jsonl(examples, out_path)
            assert os.path.exists(result)
            with open(result) as f:
                lines = f.readlines()
            assert len(lines) == 2
            record = json.loads(lines[0])
            assert record["prompt"] == "Describe image 0"
            assert record["media_type"] == "image"
        finally:
            os.unlink(out_path)

    def test_includes_embedding_preview(self):
        e = MultimodalExporter()
        ex = make_example(idx=0)
        ex.image_embedding = [0.1] * 32
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out_path = f.name
        try:
            result = e.export_jsonl([ex], out_path)
            with open(result) as f:
                record = json.loads(f.readline())
            assert "image_embedding_preview" in record
        finally:
            os.unlink(out_path)


class TestExportAll:
    def test_disabled_returns_empty(self):
        e = MultimodalExporter(config={"enabled": False})
        assert e.export_all([make_example()], "/tmp/out") == {}

    def test_exports_all_formats(self):
        e = MultimodalExporter()
        examples = [make_example(idx=i) for i in range(10)]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = e.export_all(examples, tmpdir, formats=["llava", "jsonl"])
            assert len(paths) > 0
            for key, path in paths.items():
                assert os.path.exists(path), f"{key} file not found: {path}"

    def test_handles_empty(self):
        e = MultimodalExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = e.export_all([], tmpdir)
            assert isinstance(paths, dict)


class TestSplit:
    def test_train_val_test_splits(self):
        e = MultimodalExporter(config={"train_split": 0.8, "val_split": 0.1, "test_split": 0.1})
        examples = [make_example(idx=i) for i in range(100)]
        train, val, test = e._split(examples)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_preserves_examples(self):
        e = MultimodalExporter()
        examples = [make_example(idx=i) for i in range(10)]
        train, val, test = e._split(examples)
        all_examples = train + val + test
        assert len(all_examples) == 10


class TestImageToDataUrl:
    def test_creates_data_url(self, sample_image_path):
        e = MultimodalExporter()
        url = e._image_to_data_url(sample_image_path)
        assert url.startswith("data:image/jpeg;base64,")

    def test_fallback_on_error(self):
        e = MultimodalExporter()
        url = e._image_to_data_url("nonexistent.png")
        assert url == "nonexistent.png"


class TestStats:
    def test_initial_stats(self):
        e = MultimodalExporter()
        assert e.get_stats()["exported"] == 0
        assert e.get_stats()["by_format"] == {}
