import json
import os
import random
import base64
from typing import Dict, List, Optional, Any, Tuple
from io import BytesIO

from .schema import MultimodalExample, MediaType


class MultimodalExporter:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.formats = self.config.get("formats", ["llava", "openai_vision", "blip", "jsonl"])
        self.train_split = self.config.get("train_split", 0.8)
        self.val_split = self.config.get("val_split", 0.1)
        self.test_split = self.config.get("test_split", 0.1)
        self.include_images = self.config.get("include_images", True)
        self.max_examples_per_file = self.config.get("max_examples_per_file", 10000)
        self.stats = {"exported": 0, "by_format": {}}

    def export_llava(
        self, examples: List[MultimodalExample], output_path: str
    ) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        path = output_path if output_path.endswith(".json") else output_path + ".json"

        data = []
        for ex in examples:
            if not ex.image_path or not ex.prompt:
                continue
            record = {
                "id": ex.id or str(hash(ex.prompt)),
                "image": ex.image_path,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{ex.prompt}"},
                    {"from": "gpt", "value": ex.response},
                ],
                "media_type": ex.media_type.value,
                "category": ex.category,
            }
            if ex.ocr_text:
                record["ocr_text"] = ex.ocr_text
            if ex.caption:
                record["caption"] = ex.caption
            data.append(record)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self._count("llava", len(data))
        return path

    def export_openai_vision(
        self, examples: List[MultimodalExample], output_path: str
    ) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        path = output_path if output_path.endswith(".jsonl") else output_path + "_vision.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                if not ex.image_path or not ex.prompt:
                    continue
                image_data_url = self._image_to_data_url(ex.image_path) if self.include_images else ex.image_path

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": ex.prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    },
                    {"role": "assistant", "content": ex.response},
                ]
                record = {"messages": messages, "media_type": ex.media_type.value}
                f.write(json.dumps(record) + "\n")

        self._count("openai_vision", len(examples))
        return path

    def export_blip(
        self, examples: List[MultimodalExample], output_path: str
    ) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        path = output_path if output_path.endswith(".json") else output_path + "_blip.json"

        data = []
        for ex in examples:
            if not ex.image_path:
                continue
            record = {
                "image": ex.image_path,
                "caption": ex.caption or ex.response,
                "media_type": ex.media_type.value,
            }
            if ex.ocr_text:
                record["ocr_text"] = ex.ocr_text
            if ex.metadata.get("qa_pairs"):
                record["qa_pairs"] = ex.metadata["qa_pairs"]
            data.append(record)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self._count("blip", len(data))
        return path

    def export_jsonl(
        self, examples: List[MultimodalExample], output_path: str
    ) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        path = output_path if output_path.endswith(".jsonl") else output_path + ".jsonl"

        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                record = {
                    "id": ex.id,
                    "prompt": ex.prompt,
                    "response": ex.response,
                    "image_path": ex.image_path,
                    "ocr_text": ex.ocr_text,
                    "caption": ex.caption,
                    "media_type": ex.media_type.value,
                    "modality": ex.modality.value,
                    "category": ex.category,
                    "quality_score": ex.quality_score,
                }
                if ex.image_embedding:
                    emb_str = ",".join(f"{v:.6f}" for v in ex.image_embedding[:16])
                    record["image_embedding_preview"] = f"[{emb_str},...]"
                f.write(json.dumps(record) + "\n")

        self._count("jsonl", len(examples))
        return path

    def export_all(
        self,
        examples: List[MultimodalExample],
        output_dir: str,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        if not self.enabled:
            return {}
        os.makedirs(output_dir, exist_ok=True)

        train, val, test = self._split(examples)
        formats = formats or self.formats
        paths = {}

        for fmt in formats:
            for split_name, split_data in [
                ("train", train), ("val", val), ("test", test)
            ]:
                if not split_data:
                    continue
                out_path = os.path.join(output_dir, f"{split_name}")
                if fmt == "llava":
                    p = self.export_llava(split_data, out_path)
                elif fmt == "openai_vision":
                    p = self.export_openai_vision(split_data, out_path)
                elif fmt == "blip":
                    p = self.export_blip(split_data, out_path)
                elif fmt == "jsonl":
                    p = self.export_jsonl(split_data, out_path)
                else:
                    continue
                paths[f"{split_name}_{fmt}"] = p

        self.stats["exported"] = len(examples)
        return paths

    def _split(
        self, examples: List[MultimodalExample]
    ) -> Tuple[List[MultimodalExample], List[MultimodalExample], List[MultimodalExample]]:
        shuffled = examples[:]
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * self.train_split)
        n_val = int(n * self.val_split)
        return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]

    def _image_to_data_url(self, image_path: str) -> str:
        try:
            from PIL import Image
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return image_path

    def _count(self, fmt: str, count: int):
        self.stats["by_format"][fmt] = self.stats["by_format"].get(fmt, 0) + count

    def get_stats(self) -> Dict:
        return self.stats
