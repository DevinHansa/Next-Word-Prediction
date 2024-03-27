# LSTM Next Word Prediction Model

This project implements a recurrent neural network (RNN) with Long Short-Term Memory (LSTM) cells for predicting the next word in a sequence of text. The model is trained using TensorFlow and Keras on a dataset of cricket history texts.

## Overview

The LSTM Next Word Prediction Model demonstrates the application of recurrent neural networks, specifically LSTM cells, for language modeling tasks. By training on cricket history texts, the model can generate predictions relevant to the domain of cricket. This project serves as a learning resource and can be extended for similar text generation tasks in various domains.

## Usage

- **Data Preprocessing**: Tokenize the input text data and prepare input sequences with appropriate padding.
- **Model Training**: Define and train the LSTM model using the prepared input sequences and labels.
- **Next Word Prediction**: Utilize the trained model to predict the next word in a given sequence of text.

## Dataset

The dataset used for training the model consists of cricket history texts. It provides the necessary context for the model to learn and generate coherent text predictions related to cricket.

## Model Architecture

The model architecture consists of the following layers:

1. **Embedding Layer**: Maps input sequences of words into dense vectors of fixed size, facilitating efficient learning of word representations.
2. **LSTM Layer**: Contains LSTM cells that process the input sequences and capture long-term dependencies.
3. **Dense Layer**: Computes the probability distribution over the vocabulary for the next word prediction using softmax activation.


