# Gaze n-gram LightGBM vs Markov chain for handover intent

## Motivation
The paper "Data-Driven Gaze Interaction for Adaptive Human-Robot Handovers Through Learning Markov Process Models" (2026) uses Markov process models learned from gaze data to drive adaptive handovers. This experiment is a small first test of whether a feature-based gradient boosting model can capture longer gaze patterns (such as looking back and forth between object and robot) that a first-order Markov chain cannot.

## Question
Can a class-balanced LightGBM model on gaze n-gram and back-and-forth features predict handover intent better than a first-order Markov chain classifier on simulated gaze sequences?

## Dataset
Seeded simulated gaze sequences over four areas of interest (object, robot hand, robot face, elsewhere), generated in the script with numpy. This is NOT the paper's dataset (its data is not stated), and the baseline is a generic Markov chain classifier, not the paper's method. Both classes share the same base transition chain; the handover class has longer object-to-robot back-and-forth bursts. Sequences are split once, 70/30, stratified.

## Baseline vs improved
- Baseline: one first-order Markov chain per class, classified by log-likelihood ratio with a threshold tuned on out-of-fold training scores.
- Improved: class-balanced LightGBM on bigram and trigram counts, alternation counts, dwell-time statistics and last-3 states, tuned by 3-fold CV on the training set.

Main metric: handover-class F1 on the test set; secondary: ROC-AUC and accuracy. Mean and std over seeds 42, 43, 44.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> J Lambrecht, E Wiese (2026). *Data-Driven Gaze Interaction for Adaptive Human-Robot Handovers Through Learning Markov Process Models*. … on Advanced Robotics and its Social. https://ieeexplore.ieee.org/abstract/document/11536146/

Paper author: Prof. Jens Lambrecht, Technische Universität Braunschweig (jens.lambrecht@tu-braunschweig.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
