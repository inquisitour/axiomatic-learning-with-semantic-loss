# Axiomatic Data Generator

A robust Python toolkit for generating synthetic datasets for causal reasoning tasks, specifically designed for training and evaluating machine learning models on transitivity and d-separation problems.

## 🚀 Features

- **Multiple Task Types**: Generate data for transitivity reasoning and d-separation inference
- **Configurable Complexity**: Adjustable parameters for training vs. evaluation difficulty
- **Mathematically Correct**: Implements proper d-separation algorithm with Pearl's criteria
- **Validated Output**: Automatic validation ensures all generated examples are logically consistent
- **Modular Architecture**: Clean, extensible design with separation of concerns

### Dependencies
- Python 3.7+
- No external dependencies required (uses only standard library)

## 🏗️ Architecture

The codebase is organized into four main components:

### Core Classes

| Class | Purpose |
|-------|---------|
| `GenerationConfig` | Centralized configuration management |
| `GraphGenerator` | DAG creation with topological constraints |
| `CausalReasoner` | Causal inference and d-separation logic |
| `DataValidator` | Automatic validation of generated examples |
| `AxiomaticDataGenerator` | Main orchestrator class |

### Key Improvements Over Previous Version

#### 1. **Better Architecture**
- **Modular design**: Separated concerns into specialized classes
- **Configuration system**: All parameters centralized in `GenerationConfig` dataclass
- **Type hints**: Complete type annotations throughout
- **Error handling**: Comprehensive try-catch blocks with proper logging

#### 2. **Fixed D-Separation Algorithm**
- **Proper implementation**: Uses correct d-separation rules with colliders, chains, and forks
- **Efficient path finding**: Uses BFS for undirected path discovery
- **Accurate blocking logic**: Correctly handles conditioning sets and descendant checking

#### 3. **Improved Graph Generation**
- **Guaranteed DAG property**: Uses topological ordering to ensure valid DAGs
- **Better connectivity**: Ensures all nodes are reachable
- **Proper density control**: More predictable edge density

#### 4. **Data Validation**
- **Automatic validation**: Every generated example is validated for correctness
- **Premise parsing**: Robust parsing of natural language premises
- **Logical consistency**: Ensures labels match actual causal structure

#### 5. **Better Error Handling & Logging**
- **Retry logic**: Multiple attempts for generation with fallback
- **Progress tracking**: Shows generation progress
- **Detailed logging**: Comprehensive logging for debugging

#### 6. **Performance Improvements**
- **Efficient algorithms**: Better time complexity for path finding
- **Caching potential**: Structure ready for caching frequently computed results
- **Memory management**: Proper cleanup and resource management

#### 7. **Enhanced Robustness**
- **Input validation**: Validates all parameters before processing
- **Edge case handling**: Handles degenerate cases gracefully
- **Failure recovery**: Continues generation even if some examples fail

#### 8. **Better Code Quality**
- **Documentation**: Comprehensive docstrings for all methods
- **Consistent style**: Follows Python conventions
- **Maintainability**: Clean, readable code structure

## 🚀 Quick Start

### Basic Usage

```python
from axiomatic_data_generator import AxiomaticDataGenerator, GenerationConfig

# Initialize with default configuration
generator = AxiomaticDataGenerator()

# Generate training data
transitivity_data = generator.generate_training_data(1000, "transitivity")
dsep_data = generator.generate_training_data(1000, "d-separation")

# Generate evaluation data
length_eval = generator.generate_evaluation_data(200, "length")
```

### Custom Configuration

```python
from dataclasses import dataclass

@dataclass
class CustomConfig(GenerationConfig):
    chain_length_train: tuple = (2, 4)
    chain_length_eval: tuple = (10, 20)
    node_name_length_train: tuple = (1, 2)
    max_conditioning_set_size: int = 5

generator = AxiomaticDataGenerator(CustomConfig())
```

### Command Line Usage

```bash
python axiomatic_data_generator.py
```

This will generate all default datasets:
- `transitivity_train.jsonl` (1000 examples)
- `dsep_train.jsonl` (1000 examples)
- `length_eval.jsonl` (200 examples)
- `reversed_eval.jsonl` (200 examples)
- `shuffled_eval.jsonl` (200 examples)
- `long_names_eval.jsonl` (200 examples)
- `branching_eval.jsonl` (200 examples)

## 📋 Task Types

### Transitivity Reasoning
Tests the model's ability to infer causal relationships through chains.

**Example:**
```json
{
  "premise": "A causes B. B causes C. C causes D.",
  "hypothesis": "Does A cause D?",
  "label": "Yes"
}
```

### D-Separation Inference
Tests understanding of conditional independence in causal graphs.

**Example:**
```json
{
  "premise": "X causes Y. Z causes Y. Y causes W.",
  "hypothesis": "Are X and Z d-separated given {Y}?",
  "label": "Yes"
}
```

## 🎛️ Configuration Options

### GenerationConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `node_name_length_train` | `(1, 3)` | Node name length range for training |
| `node_name_length_eval` | `(8, 10)` | Node name length range for evaluation |
| `chain_length_train` | `(3, 6)` | Chain length range for training |
| `chain_length_eval` | `(7, 15)` | Chain length range for evaluation |
| `branching_factor_train` | `(0.3, 0.6)` | Edge density for training graphs |
| `branching_factor_eval` | `(0.7, 1.2)` | Edge density for evaluation graphs |
| `max_conditioning_set_size` | `3` | Maximum size of conditioning sets |

### Evaluation Types

| Type | Description |
|------|-------------|
| `length` | Longer causal chains than training |
| `reversed` | All causal edges reversed |
| `shuffled` | Randomized order of premise statements |
| `long_names` | Longer node names |
| `branching` | Complex branched graph structures |

## 📊 Data Format

All generated data follows the JSONL format with three fields:

```json
{
  "premise": "Natural language description of causal relationships",
  "hypothesis": "Question about causal relationship or d-separation",
  "label": "Yes or No"
}
```

## 🔧 Advanced Usage

### Custom Validation

```python
# Access the validator directly
validator = generator.validator

# Validate custom examples
is_valid = validator.validate_transitivity_example(example)
```

### Graph Analysis

```python
# Access the causal reasoner
reasoner = generator.reasoner

# Check d-separation manually
edges = [("A", "B"), ("B", "C")]
is_separated = reasoner.is_d_separated(edges, "A", "C", {"B"})
