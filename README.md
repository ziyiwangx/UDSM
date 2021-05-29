# UDSM

This repository contains the code and newly-formulated datasets of our Unsupervised Feature Disentanglement of Scene and Motion (UDSM) model for video retrieval in minimally invasive surgery.

## Requirements

- python >= 3.5
- pytroch >= 1.2.0
- cv2
- scipy

## Usage

The datasets adopted is the Cholec80 dataset with prepocessing to facilitate the learning process and validation.

## Training and testing

To train and test the UDSM model, first train the feature disentanglement module with `train_disentangle.py` , and then train the LSTM module with `train_lstm.py`.

The evaluation results are saved in a new directory, containig the files of Hamming distance, Hamming rank list and the mAP results. 

## Dataset

The retrieval dataset is formulated based on the public Cholec80 dataset.
- Download Cholec80 surgical video dataset
- Convert videos to frames and downsample from 25 fps to 1fps
- Cut the original video data into shorter clips, each accounting for 20 seconds (i.e., 20 frames)
- Discard tail frames in each phase to avoid phase transitions

https://drive.google.com/drive/folders/1SfzxnxiY90ztK1FrIz30YjLI5jPu1GVv?usp=sharing
