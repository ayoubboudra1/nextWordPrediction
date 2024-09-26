# Chatbot Next Word Prediction Model

## Overview

This repository contains a Jupyter Notebook that demonstrates a simple chatbot model designed to predict the next word in a conversation. The model utilizes TensorFlow and Keras to build a basic LSTM architecture.

## Description

The model aims to predict the next word in a sentence based on previous words. While the results are not optimal due to the small size of the dataset, it performs reasonably well for simple conversations. The training data consists of conversational snippets, making it suitable for basic interaction scenarios.

## Contents

1. **Dataset Preparation**
   - Create a Kaggle directory and set up API credentials.
   - Download the dataset from Kaggle.
   - Unzip the dataset and clean up the downloaded files.

2. **Data Loading and Preprocessing**
   - Utilize Pandas for data manipulation and NumPy for numerical operations.
   - Tokenize and pad sequences to prepare the text data for the LSTM model.

3. **Model Building**
   - Construct a Sequential LSTM model using Keras.
   - Compile the model and prepare it for training.

## Requirements

To run this notebook, you will need:

- Python 3.x
- Jupyter Notebook
- TensorFlow
- Pandas
- NumPy

pip install tensorflow pandas numpy
