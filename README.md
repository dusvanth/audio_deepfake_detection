# Audio Classification using CNN-BiLSTM


                                          Deep Learning Pipeline for Speech Audio Classification</b>



                                          MFCC Feature Extraction • 1D CNN • Bidirectional LSTM • PyTorch



## Overview

This project implements an end-to-end deep learning pipeline for speech audio
classification using a hybrid **Convolutional Neural Network (CNN) and
Bidirectional Long Short-Term Memory (BiLSTM)** architecture.

The pipeline converts raw audio signals into compact **MFCC (Mel-Frequency
Cepstral Coefficient)** representations and uses deep neural networks to
learn both local spectral patterns and temporal dependencies in speech.

The implementation covers the complete machine learning workflow:

**Audio Loading → Preprocessing → MFCC Extraction → CNN → BiLSTM →
Classification → Evaluation**


## Key Features

- Built an end-to-end **audio classification pipeline** using PyTorch
- Extracted **40-dimensional MFCC features** from speech recordings
- Standardized audio inputs to **16 kHz** and fixed 4-second duration
- Designed a hybrid **1D CNN + Bidirectional LSTM** architecture
- Used CNN layers to learn local patterns from MFCC representations
- Used BiLSTM layers to capture forward and backward temporal dependencies
- Implemented reproducible train, validation, and test data splits
- Evaluated predictions using **F1-score and classification reports**


## System Architecture


                    Raw Speech Audio
                           │
                           ▼
                  Audio Preprocessing
                           │
                  ┌────────┴────────┐
                  │                 │
              16 kHz            4 Seconds
                  │                 │
                  └────────┬────────┘
                           ▼
                  MFCC Feature Extraction
                           │
                    40 MFCC Features
                           │
                           ▼
                     Normalization
                           │
                           ▼
                     1D CNN Layers
                           │
                ┌──────────┴──────────┐
                │                     │
             Conv1D                Conv1D
                │                     │
             ReLU +                ReLU +
            MaxPool               MaxPool
                │                     │
                └──────────┬──────────┘
                           ▼
                Bidirectional LSTM
                           │
                           ▼
                  Fully Connected Layer
                           │
                           ▼
                  Binary Classification
