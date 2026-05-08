import json
import re
from typing import Dict, List, Optional


class FormatNormalizer:
    @staticmethod
    def normalize_to_instruction_format(data: Dict) -> Dict:
        return {
            "instruction": data.get("instruction", data.get("prompt", data.get("question", ""))),
            "input": data.get("input", data.get("context", "")),
            "output": data.get("output", data.get("response", data.get("answer", ""))),
            "category": data.get("category", ""),
            "difficulty": data.get("difficulty", 1),
            "metadata": {k: v for k, v in data.items() if k not in
                         ["instruction", "input", "output", "category", "difficulty"]},
        }

    @staticmethod
    def normalize_conversation(messages: List[Dict]) -> Dict:
        if not messages:
            return {"instruction": "", "output": ""}

        result = {"instruction": "", "input": "", "output": ""}
        if len(messages) >= 2:
            result["instruction"] = messages[0].get("content", "") if isinstance(messages[0], dict) else str(messages[0])
            result["output"] = messages[-1].get("content", "") if isinstance(messages[-1], dict) else str(messages[-1])

        if len(messages) > 2:
            context_messages = []
            for msg in messages[1:-1]:
                role = msg.get("role", "user") if isinstance(msg, dict) else "user"
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                context_messages.append({"role": role, "content": content})
            result["input"] = json.dumps(context_messages)

        return result

    @staticmethod
    def unify_jsonl_format(input_path: str, output_path: str, format_type: str = "instruction"):
        with open(input_path, "r", encoding="utf-8") as inf, \
             open(output_path, "w", encoding="utf-8") as outf:
            for line in inf:
                data = json.loads(line.strip())
                if format_type == "instruction":
                    normalized = FormatNormalizer.normalize_to_instruction_format(data)
                elif format_type == "conversation":
                    messages = data.get("messages", data.get("conversation", []))
                    normalized = FormatNormalizer.normalize_conversation(messages)
                else:
                    normalized = data
                outf.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        text = re.sub(r'\t', '    ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    @staticmethod
    def normalize_code_blocks(text: str) -> str:
        text = re.sub(r'```(\w*)\n*```', '', text)
        text = re.sub(r'``(\w+)\n(.*?)``', r'```\1\n\2```', text, flags=re.DOTALL)
        return text
