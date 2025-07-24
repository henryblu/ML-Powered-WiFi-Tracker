# Machine Learning Scripts

This folder contains several experiments for processing CSI data with different
machine learning models. The scripts expect the CSV files under
`training_data/` as input and produce accuracy metrics or trained models.

## File overview

- `KNN_SVM_RF_ExtractedFeatures.py` - runs KNN, SVM and RandomForest classifiers
  on pre-computed CSI feature vectors.
- `KNN_SVM_RF_RawCSI.py` - similar classifiers applied directly to raw CSI
  values.
- `LSTM_ExtractedFeatures.py` - LSTM network that consumes the extracted feature
  representation.
- `LSTM_RawCSI.py` - LSTM variant that processes the raw CSI sequences.
- `NN_RawCSI.py` - small feed-forward network for raw CSI data.
- `simple_nn.py` - simplified neural network example used by the notebooks.
- `simple_nn.ipynb` - notebook form of `simple_nn.py` with exploratory cells.
- `simple_nn_multi.ipynb` - notebook showing a multi-class variant of the simple
  network.
- `nn_exploration.py` - script for quickly trying different network
  architectures and reporting their accuracy.
- `nn_exploration.ipynb` - Jupyter notebook used for debugging and interactive
  experiments.

## Usage

Most scripts expect a CSV file from the `training_data/` directory. Update the
`dataset` or `file_path` variable near the top of a script to reference your CSV
and then run it with Python:

```bash
python ML_Scripts/NN_RawCSI.py
```

Notebook files (`*.ipynb`) can be opened in Jupyter Lab for interactive
experimentation.
