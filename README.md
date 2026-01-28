# Flight Cancellation Predictor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Production-ready ML system for predicting flight cancellations using AutoGluon's TabPFNMix model.

## 🚀 Quick Start

```bash
# Install
make setup

# Run complete pipeline
make all

# Or use CLI
flight-predictor train --config configs/model_config.yaml
flight-predictor predict --input data/new_flights.csv
```

## 📊 Performance

- **ROC-AUC**: 91.87%
- **Precision**: 100%
- **Recall**: 81.82%
- **Training Time**: < 40 seconds
- **Inference**: 515 rows/s

## 📦 Installation

```bash
# Using uv (Recommended)
uv venv
uv pip install -r requirements.txt

# Using pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🛠️ Usage

### CLI Commands

```bash
# Prepare data
flight-predictor prepare-data

# Train model
flight-predictor train

# Evaluate
flight-predictor evaluate --model models/latest

# Predict
flight-predictor predict --input data/test.csv
```

### Make Commands

```bash
make help           # Show all commands
make train          # Train model
make evaluate       # Evaluate model
make predict        # Make predictions
make test           # Run tests
```

## 📝 Documentation

See [docs/](docs/) for detailed documentation.

## 👥 Maintainer

This fork is maintained by **Jun Yu** ([@junyu-boston](https://github.com/junyu-boston))

## 🤝 Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for a list of contributors.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.
