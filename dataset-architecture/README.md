# Advanced Chatbot Dataset Architecture

Production-grade dataset pipeline for training AI chatbots optimized for competitive programming, software engineering, reasoning, debugging, and conversational intelligence.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA INGESTION                          │
│  JSON │ JSONL │ CSV │ Parquet │ HuggingFace │ Synthetic     │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                             │
│  Format Normalization │ Encoding Fix │ Chat→Instruction     │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               VALIDATION & CLEANING                          │
│  Code Syntax │ Markdown │ PII Removal │ Refusal Filtering   │
│  Template HC │ Safety │ Length Checks                       │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  QUALITY SCORING                             │
│  Relevance │ Correctness │ Completeness │ Clarity           │
│  Safety │ Instruction Following │ Composite Score           │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  DEDUPLICATION                               │
│  Exact Match │ MinHash (Jaccard) │ Semantic (optional)      │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FILTERING                                 │
│  Quality Threshold │ Length Bounds │ Category Balance       │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              SYNTHETIC AUGMENTATION                          │
│  Template Generation │ Paraphrasing │ CoT Injection         │
│  Difficulty Scaling │ Error Injection │ Hard Mining         │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              CURRICULUM SCORING                              │
│  Difficulty Assessment │ ELO Ranking │ Phase Scheduling     │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     EXPORT                                   │
│  SFT │ DPO │ Axolotl │ OpenAI │ Train/Val/Test Split       │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
dataset-architecture/
├── config/                    # YAML configuration files
│   ├── categories.yaml        # 10 dataset categories & weights
│   ├── quality.yaml           # Scoring dimensions & thresholds
│   └── pipeline.yaml          # Pipeline stages & parameters
├── src/                       # Source code
│   ├── pipeline/              # Core pipeline modules
│   │   ├── ingestion.py       # Data loading (JSON, CSV, HF, etc.)
│   │   ├── preprocessing.py   # Text normalization & cleaning
│   │   ├── cleaning.py        # Refusal/PII/template removal
│   │   ├── validation.py      # Code & markdown validation
│   │   ├── export.py          # Multi-format export
│   │   └── orchestrator.py    # Pipeline orchestration
│   ├── quality/               # Quality systems
│   │   ├── scoring.py         # Multi-dimension quality scoring
│   │   ├── deduplication.py   # Exact & MinHash dedup
│   │   ├── filtering.py       # Quality-based filtering
│   │   └── ranking.py         # ELO & curriculum ranking
│   ├── synthetic/             # Data generation
│   │   ├── generator.py       # Template-based generation
│   │   ├── templates.py       # Category-specific templates
│   │   └── augmenter.py       # Paraphrasing & scaling
│   ├── curriculum/            # Curriculum learning
│   │   ├── difficulty.py      # Multi-feature difficulty scoring
│   │   └── scheduler.py       # Linear/exponential/pacing
│   ├── utils/                 # Utilities
│   │   ├── code_validator.py  # Multi-language syntax check
│   │   ├── markdown_validator.py  # Markdown quality check
│   │   └── format_normalizer.py   # Schema standardization
│   └── dataset_catalog.py     # Best open-source datasets
├── scripts/                   # Run scripts
│   ├── run_pipeline.py        # Main pipeline runner
│   ├── evaluate_quality.py    # Dataset quality analysis
│   └── generate_synthetic.py  # Synthetic data generation
├── tests/                     # Test suite
│   ├── test_validators.py     # Code & markdown validator tests
│   ├── test_cleaning.py       # Cleaning & PII removal tests
│   └── test_pipeline.py       # End-to-end pipeline tests
├── data/                      # Data directories
│   ├── raw/                   # Raw input data
│   ├── processed/             # Intermediate data
│   └── curated/               # Final curated data
├── fine_tuning_workflow.md    # Complete training protocol
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python scripts/run_pipeline.py --huggingface --output exports/my_dataset

# Generate synthetic data only
python scripts/generate_synthetic.py --count 50000 --augment

# Evaluate dataset quality
python scripts/evaluate_quality.py exports/my_dataset
```

## 10 Dataset Categories

| Category | Weight | Sources | Training Focus |
|----------|--------|---------|---------------|
| Competitive Programming | 15% | CodeContests, APPS, TACO | Algorithmic problem solving, contests |
| Algorithms & DSA | 15% | CodeAlpaca, Magicoder, LeetCode | Data structures, design patterns |
| Debugging | 12% | SWE-bench, CodeReview | Bug fixing, root cause analysis |
| System Design | 10% | Synthetic, curated | Architecture, trade-offs, scaling |
| General Reasoning | 10% | BBH, GSM8K, ARC | Logical reasoning, critical thinking |
| Math & Logic | 10% | MathDataset, MetaMathQA | Proofs, probability, statistics |
| Conversational AI | 10% | UltraChat, OASST1, ShareGPT | Dialogue, instruction following |
| Technical Docs | 6% | Doc-Code, Stack Exchange | Technical writing, explanations |
| Tool Usage | 6% | ToolBench, API-Bank | Function calling, shell, git |
| File/Image Understanding | 6% | ChartQA, DocVQA | Multi-file, diagram analysis |

## Quality Scoring Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Relevance | 25% | Response addresses the instruction |
| Correctness | 30% | Factually/technically accurate |
| Completeness | 15% | Fully answers all aspects |
| Clarity | 10% | Well-structured and clear |
| Safety | 10% | Appropriate and unbiased |
| Instruction Following | 10% | Adheres to format/constraint requirements |

## Training Protocol (3-Stage)

1. **SFT** - Supervised fine-tuning with curriculum learning
2. **DPO/ORPO** - Preference optimization from rankings
3. **Hard Mining** - Fine-tuning on difficult edge cases

See `fine_tuning_workflow.md` for complete training details.

## Best Open-Source Datasets

Key datasets integrated: CodeContests, SWE-bench, BBH, GSM8K, MetaMathQA, UltraChat, OASST1, ToolBench, ChartQA, MathDataset, APPS, TACO, Magicoder, and 30+ more (see `src/dataset_catalog.py`).

## Production Best Practices

- **Quality Gates**: Composite quality threshold of 0.65+
- **Deduplication**: Exact + MinHash (Jaccard 0.85) dedup
- **Safety**: PII removal, refusal detection, toxicity filtering
- **Validation**: AST parsing for code, structured markdown checks
- **Curriculum**: 5-phase progressive difficulty scheduling
- **Hard Mining**: Focus on low-quality/high-difficulty examples
- **Reproducibility**: Seeded RNG, checkpointing, config tracking
- **Iterative**: Self-improvement loops with quality feedback
