# Comparing Performance of Neural Network Architectures Using Data on Wildfire Propagation

### Author: Alec Creasy

## Overview

This project was completed as part of the requirements of the CSCI 6620 - Research Methods in Computer Science course at Middle Tennessee State University. The main aim of the project was to compare the Long Short-Term memory (LSTM) and Gated Recurrent Unit (GRU).

## Abstract

Machine learning techniques have been used in recent studies to predict wildfire propagation. In these endeavors, Artificial Neural Networks (ANNs) have been a popular choice to build models to solve this task. Within these models, two subsets of ANNs, Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) are a popular choice for predicting the likelihood of a fire spreading across a region over a specified number of time steps. Most existing research makes use of the Long Short-Term Memory (LSTM) variant of the RNN architecture, a special type of RNN that can retain long-term context amongst many time steps in training. The Gated Recurrent Unit (GRU) RNN is a simplified version of the LSTM architecture that hasn't been explored as much. This study explores the performance differences between the LSTM and GRU architectures on wildfire propagation prediction tasks using quantifiable metrics.

## How To Run It

### Required Software and Packages

You will need Python3 and pip installed to run the script/notebook. The required packages are included in the requirements.txt.
Once you have Python installed, you will need to install the packages. Navigate to the directory where the repository is stored on your machine, and run the following command:
```
pip install -r requirements.txt
```

This will install the required packages needed to run the script/notebook.

### Acquiring an API Key

You will need a FIRMS Map API key to run the script. You can acquire one [here](https://firms.modaps.eosdis.nasa.gov/api/map_key/). Once you have it, hold onto it. We will need it again in a moment.

### Modifying the Hyperparameters

The script can be run without Jupyter notebooks and run as a standalone Python script. The notebook is mainly useful for plotting the results and creating graphs of the models, the but the script will train the model and record the results that same way as the Jupyter notebook. For this tutorial, we will focus on the Python script.

The script is setup to run the experiment with the same data. Sequence length and batch size are changeable within the script, but before we can look at any of this, we need to define the map key. In the script, find the following code block and replace the "00000000000000000" with your map key, but **BE SURE YOU INCLUDE THE QUOTATION MARKS (*).**

```
# Define your MAP_KEY
MAP_KEY = "00000000000000000"  # Replace with your actual MAP_KEY
```

To change the sequence length, find the line where the patch sequences are generated and change the 5 in "seq_len=5" to the length of the sequence you'd like **(NOTE: the data collected only ranges from November 1, 2018-November 11, 2018, so the maximum sequence length is 9 days, where the 10th day is the predicted day!)**

```
# 2. Generate patch sequences
x_tensor, y_tensor = generate_patch_sequences(daily_grids, seq_len=5, patch_size=3)
```

From here, you can define different hyperparameters such as the batch size and time step length. To change the batch size, find the line where the DataModule is created, and replace the 32 in "batch_size=32" with the batch size you'd like. In the original experiment, we use a batch size of 32.

```
# 3. Create DataModule and set it up.
data_module = FireDataModule(x_tensor, y_tensor, batch_size=32)
data_module.setup()
```

The remaining hyperparameters can be changed by modifying the architecture of the models directly if you so wish, but the main setup is 3 ConvLSTM/ConvGRU layers with 64, 128, and 256 channels each with a kernel size of 3x3 with padding (no pooling) and a learning rate of 0.001.

The default paths for the logs for "logs/LSTM-Log/lstm/metrics.csv" and "logs/GRU-Log/gru/metrics.csv" for the ConvLSTM and ConvGRU blocks, respectively. If you'd like to change this, change the corresponding directory names in the 2 different logger blocks. The first block refers to the ConvLSTM mdoel while the second refers to the ConvGRU model.

```
logger = pl.loggers.CSVLogger("logs",
                              name="LSTM-Log",
                             version="lstm")

logger = pl.loggers.CSVLogger("logs",
                              name="GRU-Log",
                             version="gru")
```

### Running the Script

Now that we have all of the steps above done, we can now run the script which will train both models and print the results in the logs you specified above. Open a terminal window and navigate to the directory of the repository. Once there, type the following command:

```
python main.py
```

If you receive an error that the "python" command is not found, try using "python3" instead. If you still receive an error, you may need to reinstall Python or ensure that it is in your system's PATH.

And that's it! The models should now be training for your experiment
