# Machine Learning Repository with PyTorch

This is a machine learning repository containing PyTorch-based tutorials and projects using Jupyter notebooks for hands-on learning of neural networks, data preprocessing, and model training.

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Initial Setup and Dependencies
- Install all Python dependencies:
  - `pip3 install -r requirements.txt` -- takes 2.5 minutes. NEVER CANCEL. Set timeout to 5+ minutes.
- Verify installations work:
  - `python3 -c "import torch; print('PyTorch version:', torch.__version__)"`
  - `python3 -c "import pandas as pd; print('Pandas version:', pd.__version__)"`
  - `python3 -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"`
  - `python3 -c "import torchmetrics; print('Torchmetrics version:', torchmetrics.__version__)"`

### Running Jupyter Notebooks
- Execute notebooks in headless mode for testing:
  - `jupyter nbconvert --to notebook --execute practiceProj/lessons/introToPyTorch.ipynb --output /tmp/test_intro_pytorch.ipynb`
  - `jupyter nbconvert --to notebook --execute practiceProj/lessons/BinaryModel.ipynb --output /tmp/test_binary_model.ipynb`
  - `jupyter nbconvert --to notebook --execute practiceProj/water_potability/water_potability.ipynb --output /tmp/test_water_potability.ipynb` -- takes 15 seconds. NEVER CANCEL. Set timeout to 60+ seconds.
- Start interactive Jupyter server:
  - `jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser`
  - Server will be accessible at: `http://127.0.0.1:8888/lab?token=<generated_token>`
  - Token will be displayed in the startup output
- Convert notebooks to Python scripts for automated testing:
  - `jupyter nbconvert --to python practiceProj/lessons/introToPyTorch.ipynb --output /tmp/intro_script.py`

### Testing and Validation
- Test basic PyTorch functionality:
```python
python3 -c "
import torch
import torch.nn as nn
print('PyTorch version:', torch.__version__)
tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print('Basic tensor operations work:', tensor.shape)
model = nn.Linear(2, 1)
output = model(tensor)
print('Neural network creation works:', output.shape)
"
```

- Test water potability ML workflow (complete end-to-end validation):
```python
python3 -c "
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy

# Load and validate data
data = pd.read_csv('./practiceProj/water_potability/water_potability.csv')
print('Data loaded. Shape:', data.shape)

# Quick model training test
data = data.fillna(data.mean())
features = data.drop('Potability', axis=1)
target = data['Potability']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Simple validation that the full pipeline works
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)
print('Pipeline test completed successfully')
"
```

## Validation Scenarios

**ALWAYS manually validate changes by running these complete scenarios after making modifications:**

### Scenario 1: Basic PyTorch Tutorial Validation
1. Run the introToPyTorch notebook: `jupyter nbconvert --to notebook --execute practiceProj/lessons/introToPyTorch.ipynb --output /tmp/test_intro.ipynb`
2. Verify tensor operations, stacking, and basic neural network concepts work
3. Check that PyTorch version 2.8.0+ is being used

### Scenario 2: Binary Model Tutorial Validation  
1. Run the BinaryModel notebook: `jupyter nbconvert --to notebook --execute practiceProj/lessons/BinaryModel.ipynb --output /tmp/test_binary.ipynb`
2. Verify nn.Linear layers, activation functions, and binary classification concepts work
3. Test model creation and forward pass functionality

### Scenario 3: Complete ML Project Validation
1. Run the full water potability notebook: `jupyter nbconvert --to notebook --execute practiceProj/water_potability/water_potability.ipynb --output /tmp/test_water.ipynb` -- takes 15 seconds. NEVER CANCEL.
2. Verify data loading from CSV (3276 rows, 10 columns)
3. Verify data preprocessing (missing value handling, feature scaling)
4. Verify model training (neural network with multiple layers)
5. Verify model evaluation (accuracy calculation with torchmetrics)
6. Expected final accuracy should be around 0.60-0.70 range

## Timing Expectations and Timeouts

- **Dependency installation**: 2.5 minutes. Set timeout to 5+ minutes. NEVER CANCEL.
- **Individual notebook execution**: 
  - `introToPyTorch.ipynb`: 6-7 seconds. Set timeout to 60+ seconds.
  - `BinaryModel.ipynb`: 5-6 seconds. Set timeout to 60+ seconds.
  - `water_potability.ipynb`: 14-15 seconds. Set timeout to 60+ seconds. NEVER CANCEL.
- **Basic PyTorch tests**: Under 1 second each
- **Complete ML workflow validation**: 1-2 seconds for quick test, 14-15 seconds for full notebook
- **Jupyter server startup**: 3-5 seconds to be ready, accessible at `http://127.0.0.1:8888/lab`

## Key Projects and Components

### Repository Structure
```
practiceProj/
├── lessons/
│   ├── introToPyTorch.ipynb     # PyTorch tensor basics and operations
│   ├── BinaryModel.ipynb        # Neural network fundamentals  
│   └── sigmoid.png             # Reference diagram
└── water_potability/
    ├── water_potability.ipynb   # Complete ML project
    └── water_potability.csv     # Dataset (3276 rows, 10 features)
```

### Core Dependencies
- **PyTorch 2.8.0+**: Deep learning framework for neural networks
- **Pandas 2.3.2+**: Data manipulation and CSV loading  
- **Scikit-learn 1.7.1+**: Data preprocessing (StandardScaler, train_test_split)
- **Torchmetrics 1.8.1+**: Model evaluation metrics (Accuracy)
- **Jupyter**: Interactive notebook environment
- **NumPy, Matplotlib**: Supporting libraries for data science

### Important Concepts Covered
- **Tensor Operations**: Creation, manipulation, stacking, reshaping
- **Neural Networks**: Linear layers, activation functions (ReLU, Sigmoid)  
- **Data Preprocessing**: Missing value handling, feature scaling, train/test splits
- **Model Training**: Forward pass, backward pass, optimization (Adam)
- **Model Evaluation**: Binary classification metrics, accuracy calculation

## Common Tasks Reference

### Quick Commands for Development
```bash
# Install environment
pip3 install -r requirements.txt

# Test all notebooks work
jupyter nbconvert --to notebook --execute practiceProj/lessons/introToPyTorch.ipynb --output /tmp/test1.ipynb
jupyter nbconvert --to notebook --execute practiceProj/lessons/BinaryModel.ipynb --output /tmp/test2.ipynb  
jupyter nbconvert --to notebook --execute practiceProj/water_potability/water_potability.ipynb --output /tmp/test3.ipynb

# Quick validation test
python3 -c "import torch, pandas, sklearn, torchmetrics; print('All imports successful')"
```

### Files You'll Frequently Reference
- `practiceProj/water_potability/water_potability.ipynb`: Complete ML project example
- `practiceProj/lessons/introToPyTorch.ipynb`: PyTorch fundamentals
- `practiceProj/water_potability/water_potability.csv`: Sample dataset for experimentation
- `requirements.txt`: All necessary Python dependencies

### Data Format Notes
- Water potability CSV has 9 features + 1 target (Potability: 0/1)  
- Features: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity
- Contains missing values that need preprocessing
- Target is binary classification (potable vs non-potable water)

## Development Guidelines

### When Making Changes
- ALWAYS run the validation scenarios after any code modifications
- ALWAYS test that notebooks still execute successfully with `jupyter nbconvert --execute`
- Pay attention to tensor shape mismatches (common issue with torchmetrics)
- Ensure model outputs are properly shaped for loss functions and metrics
- Use `.squeeze()` or `.view()` to fix tensor dimension issues when needed

### Performance Notes
- PyTorch uses CPU by default in this environment (CUDA available: False)
- Model training is fast due to small dataset size (3276 samples)
- Batch size of 32 works well for the water potability dataset
- 5-20 epochs usually sufficient for convergence on this dataset

### No Build System
- This is a Python notebook-based project with no compilation step
- No testing framework (pytest, unittest) is configured
- No linting tools (black, flake8) are configured  
- Changes are validated by running notebooks and manual testing