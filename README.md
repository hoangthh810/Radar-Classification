# A Lightweight CNN Architecture for Automatic Radio Frequency Signal Recognition

## Abstract
This project develops an automated system for identifying Radio Frequency (RF) and radar signals, a critical task in modern spectrum monitoring. By transforming raw RF signals into time-frequency spectrogram images, the recognition problem is effectively addressed using computer vision techniques. We propose a lightweight Convolutional Neural Network (CNN) architecture optimized with fewer than 100,000 trainable parameters, ensuring feasibility for deployment on resource-constrained edge devices. The model was trained and evaluated on a diverse dataset of 12 signal classes, achieving high accuracy while strictly adhering to technical constraints regarding model architecture and system output formats.

## Download Dataset
* **Training Set:** [Kaggle - Radar Common Signal Data Train](https://www.kaggle.com/datasets/hoangcat/radar-common-signal-data-train)
* **Test Set:** [Kaggle - Radar Common Signal Data Test](https://www.kaggle.com/datasets/hoangcat/radar-common-signal-data-test)

## Datasets
The dataset consists of 12 different radio frequency signal classes, represented as $224 \times 224$ grayscale spectrogram images.

### Signal Samples
![12 Signal Classes Preview](img/classes_preview.png)
