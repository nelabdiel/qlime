# Q-LIME: Quantum LIME for Text Explanations

This repository contains the implementation of **Q-LIME Pi (Quantum LIME)**, a quantum-inspired extension of LIME (Local Interpretable Model-agnostic Explanations), applied to text-based machine learning tasks. Q-LIME leverages quantum-inspired methods to explore the impact of flipping binary features on a model's output more efficiently.

## Features

### 1. Classical and Quantum LIME Comparison
- **Classical LIME**: Perturbs features to evaluate their impact on model predictions using a standard implementation.
- **Q-LIME Pi**: Uses quantum-inspired binary encoding and flipping to compute feature importance.

### 2. Feature Visualization
- Horizontal bar graph visualization for feature contributions.
- Color-coded feature importance visualization directly on the text using HTML in Jupyter Notebooks.

### 3. Performance Evaluation
- Benchmark comparisons between Classical LIME and Q-LIME Pi.
- Metrics include runtime, accuracy, and overlap in top features.

---

## Installation

### Prerequisites
- Python 3.11+
- Libraries:
  - `numpy`
  - `pennylane`
  - `scikit-learn`
  - `matplotlib`
  - `lime`
  - `IPython`

Install the required libraries using:
```bash
pip install numpy pennylane scikit-learn matplotlib lime ipython
```

### Dataset
This project uses the IMDb sentiment analysis dataset. Download the dataset from [IMDb Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) and place it in the `./aclImdb/` directory.

---

## Usage

### 1. Running the Example with Q-LIME
This script runs Q-LIME on a random sample of the IMDb dataset, visualizes feature contributions as a bar graph, and highlights the top contributing words directly on the text.
```python
python main.py
```

### 2. Classical and Quantum LIME Comparison
The comparison script measures the runtime and feature overlap between Classical LIME and Q-LIME for several text samples. Adjust parameters such as `max_features` to test different configurations.

### 3. Benchmarking Performance
The benchmark script evaluates both methods across varying numbers of features, reporting accuracy, runtime, and overlap in top feature rankings.

---

## Files

### `highlighting.py`
Contains the implementation of Q-LIME with:
- Horizontal bar graph visualization of feature contributions.
- Color-coded text highlighting for top features in Jupyter Notebook.

### `comparison.py`
Compares Classical LIME and Q-LIME:
- Measures runtime and top feature overlap.
- Prints top-5 features for both methods.

### `benchmark.py`
Benchmarks the performance of Classical LIME and Q-LIME:
- Reports metrics for varying feature dimensions and dataset sizes.

---

## Example Output

### Q-LIME Visualization
**Bar Graph for Feature Contributions**
![Bar Graph Example](example_bar_graph.png)

**Text Highlighting**
Highlighted text with top contributing words color-coded (green for positive, red for negative).

---

## Citation
If you use this repository in your work, please cite as:
```
@article{qlime2024,
  title={Q-LIME: Quantum LIME for Text Explanations},
  author={Nelson A. Colon Vargas},
  year={2024},
  url={https://github.com/nelabdiel/qlime}
}
```

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
