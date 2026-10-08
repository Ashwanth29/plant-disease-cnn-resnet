# Plant Disease Classification: CNN vs ResNet

## Problem Statement

A plant-disease classification system becomes difficult to train
when additional CNN layers are introduced.

This project compares a conventional CNN and a ResNet-based model
using the same plant disease dataset.

## Objectives

- Develop a conventional CNN.
- Develop a ResNet-based classifier.
- Compare training behaviour.
- Compare classification performance.
- Analyze the effect of residual connections.

## Models

### Conventional CNN

Sequential convolutional layers followed by classification.

### ResNet

Convolutional layers with residual/skip connections.

## Experimental Setup

Both models use:

- Same dataset
- Same image resolution
- Same train/test split
- Same optimizer
- Same learning rate
- Same number of epochs

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Residual Connection

The ResNet block learns:

F(x) + x

The shortcut allows information and gradients to propagate
more effectively through deeper networks.

## Conclusion

The experiment demonstrates how residual connections can improve
the optimization and learning behaviour of deeper CNN architectures.