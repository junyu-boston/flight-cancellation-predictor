# Project Organization

## 📁 Clean Project Structure

```
flight-cancellation-predictor/
│
├── 📄 Core Files (Root)
│   ├── README.md              # Main documentation
│   ├── LICENSE                # MIT License
│   ├── CONTRIBUTING.md        # Contribution guidelines
│   ├── NOTICE                 # Third-party notices
│   ├── setup.py               # Package installation
│   ├── pyproject.toml         # Modern Python config
│   ├── requirements.txt       # Production dependencies
│   ├── requirements-dev.txt   # Development dependencies
│   ├── Makefile              # Command shortcuts
│   ├── Dockerfile            # Container definition
│   ├── .dockerignore         # Docker ignore rules
│   ├── .gitignore            # Git ignore rules
│   └── .env.example          # Environment template
│
├── 📦 src/                   # Source Code (Production)
│   ├── cli/                  # Command-line interface
│   ├── data/                 # Data loading & validation
│   ├── features/             # Feature engineering
│   ├── models/               # Model training
│   ├── evaluation/           # Model evaluation
│   └── utils/                # Utility functions
│
├── ⚙️ configs/               # Configuration Files
│   ├── model_config.yaml     # Model settings
│   ├── data_config.yaml      # Data pipeline
│   ├── training_config.yaml  # Training parameters
│   └── logging_config.yaml   # Logging setup
│
├── 📊 data/                  # Data Storage
│   ├── raw/                  # Original data files
│   ├── processed/            # Processed datasets
│   ├── features/             # Engineered features
│   └── external/             # External datasets
│
├── 🤖 models/                # Model Storage
│   ├── artifacts/            # Trained models
│   ├── checkpoints/          # Training checkpoints
│   └── registry/             # Model versions
│
├── 📓 notebooks/             # Jupyter Notebooks
│   ├── exploratory/          # EDA & exploration
│   ├── experiments/          # Model experiments
│   └── reports/              # Final analyses
│
├── 📝 docs/                  # Documentation
│   ├── api/                  # API reference
│   ├── guides/               # User guides
│   ├── technical/            # Technical specs
│   └── examples/             # Code examples
│
├── 🧪 tests/                 # Test Suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── fixtures/             # Test data
│
├── 🔧 scripts/               # Utility Scripts
│   └── train.py              # Main pipeline
│
├── 📈 results/               # Analysis Results
│   └── (generated outputs)
│
├── 📁 archive/               # Old Files (Preserved)
│   ├── original_notebooks/   # Original .ipynb files
│   └── original_scripts/     # Original .py scripts
│
└── 📋 logs/                  # Application Logs
    └── (runtime logs)
```

## 🎯 Organization Principles

### 1. **Separation of Concerns**
- Each directory has a single, clear purpose
- Source code separated from configuration
- Data separated from code
- Tests isolated from production code

### 2. **Clean Root Directory**
- Only essential files in root
- Configuration files for tools (setup.py, Dockerfile, etc.)
- Documentation (README, LICENSE)
- No code files or notebooks

### 3. **Hierarchical Structure**
- `src/` - All production code
- `configs/` - All configuration
- `data/` - All data files
- `models/` - All model artifacts
- `tests/` - All testing code

### 4. **Archive Strategy**
- Old notebooks and scripts preserved in `archive/`
- Keeps history while maintaining clean structure
- Easy to reference or restore if needed

## 🚀 Quick Navigation

| What You Want | Where to Find It |
|--------------|------------------|
| Run the code | `src/` or use `make` commands |
| Change settings | `configs/*.yaml` |
| View documentation | `docs/` or `README.md` |
| Run tests | `tests/` or `make test` |
| Original notebooks | `archive/original_notebooks/` |
| Trained models | `models/artifacts/` |
| Raw data | `data/raw/` |
| Results/outputs | `results/` |

## 📋 File Categories

### Essential Files (Root)
- **Documentation**: README.md, LICENSE, CONTRIBUTING.md
- **Package Setup**: setup.py, pyproject.toml, requirements*.txt
- **Automation**: Makefile, Dockerfile
- **Configuration**: .gitignore, .dockerignore, .env.example

### Production Code (src/)
- **Modular**: Each module has single responsibility
- **Importable**: Can be used as package
- **Testable**: Clear interfaces for testing

### Configuration (configs/)
- **YAML Format**: Human-readable
- **Separated**: Different aspects in different files
- **Version Controlled**: Track changes over time

### Data Organization (data/)
- **Raw**: Original, untouched data
- **Processed**: Cleaned and transformed
- **Features**: Engineered features
- **External**: Third-party data

## ✅ Benefits of This Structure

1. **Professional**: Industry-standard layout
2. **Scalable**: Easy to add new features
3. **Maintainable**: Clear where everything belongs
4. **Collaborative**: Team members know where to look
5. **Deployable**: Ready for production
6. **Testable**: Clear test structure
7. **Documentable**: Organized documentation

## 🔧 Common Tasks

```bash
# Start working
make setup

# Train model
make train

# Run tests
make test

# View documentation
open docs/

# Check old notebooks
ls archive/original_notebooks/

# Deploy
docker build -t flight-predictor .
```

This organization follows software engineering best practices and makes the project professional, maintainable, and scalable.