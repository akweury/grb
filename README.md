# Gestalt Pattern Reasoning Benchmark

## Overview
This repository contains a dataset and benchmarking framework for **Gestalt pattern reasoning**. The dataset consists of thousands of procedurally generated visual patterns based on **Gestalt principles**, including proximity, similarity, closure, symmetry, and continuity. The benchmark is designed to evaluate both human and AI performance in recognizing and reasoning about these patterns.

## File Structure
```
gestalt_benchmark/
│── data/
│   │── raw_patterns/         # Unprocessed/generated raw patterns
│   │   │── proximity/
│   │   │   │── pattern_0001/
│   │   │   │   │── positive/
│   │   │   │   │   │── img_0001.png
│   │   │   │   │   │── img_0002.png
│   │   │   │   │── negative/
│   │   │   │   │   │── img_0001.png
│   │   │   │   │   │── img_0002.png
│   │   │── similarity/
│   │   │── closure/
│   │   │── symmetry/
│   │   │── continuity/
│   │── processed_patterns/    # Processed & labeled patterns
│   │── metadata/              # Metadata and descriptions
│   │   │── pattern_metadata.json
│
│── scripts/
│   │── proximity/             # Folder containing scripts for proximity patterns
│   │   │── pattern_0001.py
│   │   │── pattern_0002.py
│   │── similarity/            # Folder containing scripts for similarity patterns
│   │   │── pattern_0001.py
│   │   │── pattern_0002.py
│   │── closure/               # Folder containing scripts for closure patterns
│   │   │── pattern_0001.py
│   │   │── pattern_0002.py
│   │── symmetry/              # Folder containing scripts for symmetry patterns
│   │   │── pattern_0001.py
│   │   │── pattern_0002.py
│   │── continuity/            # Folder containing scripts for continuity patterns
│   │   │── pattern_0001.py
│   │   │── pattern_0002.py
│   │── utils/                 # Folder containing utility functions
│   │   │── image_processing.py
│   │   │── dataset_helpers.py
│   │── generate_patterns.py   # Script to generate patterns
│   │── process_patterns.py    # Cleaning, labeling, augmenting
│   │── evaluate_models.py     # Benchmarking models
│
│── benchmarks/
│   │── model_results/         # AI model performance results
│   │   │── model1.json
│
│── configs/
│   │── dataset_config.yaml    # Configuration file for pattern generation
│   │── model_config.yaml      # Model evaluation settings
│
│── notebooks/
│   │── pattern_analysis.ipynb # Jupyter notebooks for visualization
│   │── model_evaluation.ipynb
│
│── README.md
│── requirements.txt
```

## Installation
To use this benchmark, first clone the repository and install dependencies:
```bash
git clone https://github.com/your-repo/gestalt_benchmark.git
cd gestalt_benchmark
pip install -r requirements.txt
```

## Pattern Generation
To generate patterns based on Gestalt principles, run:
```bash
python scripts/generate_patterns.py --config configs/dataset_config.yaml
```
Generated patterns will be saved in `data/raw_patterns/`.

## Data Processing
To normalize, augment, and categorize patterns:
```bash
python scripts/process_patterns.py
```
Processed patterns will be stored in `data/processed_patterns/`.

## Benchmarking AI Models
To evaluate AI models on the dataset:
```bash
python scripts/evaluate_models.py --config configs/model_config.yaml
```
Results will be saved in `benchmarks/model_results/`.

## Gestalt Principles and Patterns
The benchmark includes five **Gestalt principles**:
- **Proximity**
- **Similarity**
- **Closure**
- **Symmetry**
- **Continuity**

For each principle, there are approximately **100 patterns**. Each pattern includes:
- **50 positive images** and **50 negative images** for training.
- **50 positive images** and **50 negative images** for testing.

Patterns are generated using basic objects such as:
- **Triangle**
- **Circle**
- **Square**

Each pattern has its own folder within the respective principle directory, containing **positive** and **negative** subdirectories. Additionally, each principle has its own folder in the `scripts/` directory, and each pattern has its own script file for generation.

## Metadata Format
Each pattern has an associated metadata entry in `data/metadata/pattern_metadata.json`:
```json
{
  "pattern_0001": {
    "type": "proximity",
    "difficulty": "easy",
    "resolution": "512x512",
    "generation_parameters": {
      "shape": "circle",
      "spacing": "small",
      "alignment": "grid"
    }
  }
}
```

## Contribution
We welcome contributions to improve the dataset and evaluation framework. Please submit pull requests with explanations of changes.

## License
This project is licensed under the MIT License.

## Contact
For questions, reach out via [your contact email] or open an issue on GitHub.

---
🚀 **Ready to challenge AI with Gestalt patterns? Start now!**

