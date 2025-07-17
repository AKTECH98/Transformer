# Transformer_Scratch

A modular implementation of the Transformer architecture from scratch in PyTorch.

## Project Structure

```
Transformer_Scratch/
│
├── transformer/                # Main package for all model code
│   ├── __init__.py
│   ├── attention.py
│   ├── embedding_positional.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── transformer.py
│   ├── encoder_decoder.py
│   └── feed_forward.py
│
├── scripts/                    # For training, evaluation, and visualization scripts
│   └── positional_encoding_visualization.py
│
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   └── test_transformer.py
│
├── requirements.txt            # Dependencies (to be created)
├── README.md                   # Project overview
└── .gitignore
```

## Usage

- Import modules from the `transformer` package in your scripts or notebooks.
- Run scripts from the `scripts/` directory for visualization or experiments.
- Run tests from the `tests/` directory to verify implementation.

## Requirements

- Python 3.8+
- torch
- numpy
- matplotlib (for visualization)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## License

MIT
