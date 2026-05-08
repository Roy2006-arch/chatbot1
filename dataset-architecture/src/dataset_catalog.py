from typing import Dict, List


BEST_OPEN_SOURCE_DATASETS: Dict[str, List[Dict]] = {
    "competitive_programming": [
        {"name": "CodeContests", "source": "deepmind/code_contests", "size": "~13K", "use": "CP problems with solutions in multiple languages"},
        {"name": "Codeforces Dataset", "source": "open-r1/codeforces", "size": "~100K", "use": "Codeforces problems, submissions, and ratings"},
        {"name": "TACO", "source": "TACO-org/taco", "size": "~26K", "use": "Competitive programming with execution traces"},
        {"name": "APPS", "source": "codeparrot/apps", "size": "~10K", "use": "Programming problems for automated solution generation"},
        {"name": "USACO", "source": "open-r1/usaco", "size": "~2K", "use": "USACO contest problems with solutions"},
    ],
    "algorithms_dsa": [
        {"name": "CodeAlpaca", "source": "sahil2801/CodeAlpaca-20k", "size": "~20K", "use": "Code generation instructions"},
        {"name": "Magicoder", "source": "ise-uiuc/Magicoder_OSS_Instruct_75K", "size": "~75K", "use": "Open-source code instruction data"},
        {"name": "GeeksForGeeks", "source": "huggingface.co/datasets/greenger/GFG", "size": "~30K", "use": "DSA problems and solutions"},
        {"name": "LeetCode Solutions", "source": "greenger/leetcode-solutions", "size": "~3K", "use": "LeetCode problem solutions"},
        {"name": "CodeForce Python", "source": "patil-suraj/codeforce-python", "size": "~10K", "use": "CodeForce solutions in Python"},
    ],
    "debugging": [
        {"name": "SWE-bench", "source": "princeton-nlp/SWE-bench", "size": "~2.3K", "use": "Real-world software engineering issues"},
        {"name": "Debugging Instructions", "source": "microsoft/debugging_instructions", "size": "~25K", "use": "Buggy code with fix explanations"},
        {"name": "BugsInPy", "source": "sohailahmedkhan/BugsInPy", "size": "~500", "use": "Real Python bugs from open-source projects"},
        {"name": "Defects4J", "source": "rjust/defects4j", "size": "~800", "use": "Java bug dataset for testing"},
        {"name": "CodeReview", "source": "microsoft/codereview", "size": "~10K", "use": "Code review comments and fixes"},
    ],
    "system_design": [
        {"name": "System Design Interview", "source": "system-design-interview", "size": "~5K", "use": "Common system design interview Q&A"},
        {"name": "Awesome System Design", "source": "awesome-system-design", "size": "~3K", "use": "Curated system design resources"},
        {"name": "Distributed Systems Reading", "source": "dist-sys-reading", "size": "~2K", "use": "Distributed systems knowledge"},
        {"name": "Architecture Decision Records", "source": "adr", "size": "~1K", "use": "Real architecture decisions"},
    ],
    "general_reasoning": [
        {"name": "BigBench Hard (BBH)", "source": "lukaemon/bbh", "size": "~6.5K", "use": "Challenging reasoning tasks"},
        {"name": "GSM8K", "source": "openai/gsm8k", "size": "~8.5K", "use": "Grade school math word problems"},
        {"name": "ARC-Challenge", "source": "allenai/ai2_arc", "size": "~2.6K", "use": "Science exam questions requiring reasoning"},
        {"name": "StrategyQA", "source": "google/strategy_qa", "size": "~2.8K", "use": "Multi-step strategy reasoning"},
        {"name": "Logical Reasoning", "source": "cais/common-reasoning", "size": "~5K", "use": "Logical deduction and reasoning"},
    ],
    "math_logic": [
        {"name": "Math Dataset", "source": "math_dataset/math_dataset", "size": "~2M", "use": "Mathematics problems across topics"},
        {"name": "MetaMathQA", "source": "meta-math/MetaMathQA", "size": "~395K", "use": "Math problems with augmented reasoning paths"},
        {"name": "MMLU Math", "source": "mmlu/math", "size": "~1K", "use": "College-level math questions"},
        {"name": "ProofNet", "source": "openai/proofnet", "size": "~15K", "use": "Theorem proving problems"},
        {"name": "NuminaMath-CoT", "source": "AI-MO/NuminaMath-CoT", "size": "~860K", "use": "Math with chain-of-thought reasoning"},
        {"name": "DAMP", "source": "damp/math_problems", "size": "~200K", "use": "Diverse math problems with solutions"},
    ],
    "conversational_ai": [
        {"name": "UltraChat", "source": "HuggingFaceH4/ultrachat_200k", "size": "~200K", "use": "Synthetic multi-turn conversations"},
        {"name": "OpenAssistant (OASST1)", "source": "OpenAssistant/oasst1", "size": "~88K", "use": "Human-assistant conversation tree data"},
        {"name": "ShareGPT", "source": "RyokoAI/ShareGPT52K", "size": "~52K", "use": "Real user-ChatGPT conversations"},
        {"name": "Dolly", "source": "databricks/databricks-dolly-15k", "size": "~15K", "use": "Human-generated instruction data"},
        {"name": "Anthropic HH-RLHF", "source": "anthropic/hh-rlhf", "size": "~170K", "use": "Helpful & harmless preference data"},
        {"name": "LMSys Chat", "source": "lmsys/lmsys-chat-1m", "size": "~1M", "use": "Real-world chatbot conversations"},
        {"name": "Self-Instruct", "source": "yizhongw/self_instruct", "size": "~52K", "use": "Self-generated instruction data"},
    ],
    "technical_documentation": [
        {"name": "Doc-Code", "source": "doc-code/doc-code", "size": "~50K", "use": "Documentation-code alignment pairs"},
        {"name": "GitHub CodeDoc", "source": "bigcode/code-documentation", "size": "~100K", "use": "Code with documentation strings"},
        {"name": "WikiHow", "source": "wikihow/wikihow", "size": "~200K", "use": "Instructional how-to articles"},
        {"name": "Stack Exchange", "source": "flax-sentence-embeddings/stackexchange_xml", "size": "~10M", "use": "Q&A with technical explanations"},
        {"name": "Python Docs", "source": "python/docs", "size": "~30K", "use": "Python official documentation"},
    ],
    "tool_usage": [
        {"name": "ToolBench", "source": "sambanova/ToolBench", "size": "~50K", "use": "Tool-calling conversations"},
        {"name": "API-Bank", "source": "AlibabaResearch/API-Bank", "size": "~20K", "use": "API usage dialogue data"},
        {"name": "Shell Scripts", "source": "bigcode/shell-scripts", "size": "~30K", "use": "Shell command examples"},
        {"name": "Func-Calling", "source": "nvidia/function-calling", "size": "~15K", "use": "Function calling training data"},
        {"name": "Tool-Learning", "source": "thunlp/ToolLearning", "size": "~10K", "use": "Tool use learning data"},
    ],
    "file_image_understanding": [
        {"name": "ChartQA", "source": "HuggingFaceM4/ChartQA", "size": "~30K", "use": "Chart understanding Q&A"},
        {"name": "Code File Analysis", "source": "custom", "size": "~10K", "use": "Multi-file code understanding"},
        {"name": "DocVQA", "source": "russ048/DocVQA", "size": "~50K", "use": "Document visual Q&A"},
        {"name": "Diagram Understanding", "source": "custom", "size": "~5K", "use": "Architecture diagram analysis"},
    ],
}


PREFERENCE_AND_RANKING_DATASETS = [
    {"name": "UltraFeedback", "source": "openbmb/UltraFeedback", "size": "~64K", "use": "Fine-grained quality feedback on model outputs"},
    {"name": "HelpSteer", "source": "nvidia/HelpSteer", "size": "~36K", "use": "Multi-dimensional human preference ratings"},
    {"name": "CodeFeedback", "source": "openbmb/CodeFeedback", "size": "~10K", "use": "Code quality preference data"},
    {"name": "Math-Shepherd", "source": "peiyi9979/Math-Shepherd", "size": "~445K", "use": "Process-level reward for math reasoning"},
    {"name": "PRM800K", "source": "openai/prm800k", "size": "~800K", "use": "Step-level correctness labels for math solutions"},
    {"name": "Evol-Instruct", "source": "WizardLM/Evol-Instruct-70k", "size": "~70K", "use": "Evolved instruction complexity data"},
]


def get_dataset_summary() -> str:
    lines = ["# Dataset Catalog Summary\n"]
    total = 0
    for category, datasets in BEST_OPEN_SOURCE_DATASETS.items():
        cat_total = sum(int(ds["size"].replace("~", "").replace("K", "000").replace("M", "000000").split()[0]) for ds in datasets)
        lines.append(f"- **{category.replace('_', ' ').title()}**: {len(datasets)} datasets, ~{cat_total:,} total examples")
        total += cat_total

    pref_total = sum(int(ds["size"].replace("~", "").replace("K", "000").replace("M", "000000").split()[0]) for ds in PREFERENCE_AND_RANKING_DATASETS)
    lines.append(f"- **Preference & Ranking**: {len(PREFERENCE_AND_RANKING_DATASETS)} datasets, ~{pref_total:,} total examples")
    total += pref_total

    lines.append(f"\n**Grand total: ~{total:,} examples across 10 categories**")
    return "\n".join(lines)


def get_datasets_for_category(category: str) -> List[Dict]:
    return BEST_OPEN_SOURCE_DATASETS.get(category, [])


def get_all_recommended_datasets() -> Dict[str, List[Dict]]:
    return BEST_OPEN_SOURCE_DATASETS.copy()
