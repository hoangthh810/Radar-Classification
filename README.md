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

## Data Statistics
| Class | snr01 | snr02 | snr03 | snr04 | snr05 | snr06 | snr07 | snr08 | **Total Instances** | **Image Size** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **16-QAM** | 858 | 769 | 796 | 843 | 789 | 798 | 758 | 789 | **6,400** | $224 \times 224$ |
| **B-FM** | 778 | 761 | 791 | 823 | 832 | 789 | 785 | 841 | **6,400** | $224 \times 224$ |
| **BPSK** | 828 | 781 | 812 | 838 | 793 | 783 | 788 | 777 | **6,400** | $224 \times 224$ |
| **Barker** | 789 | 816 | 751 | 819 | 822 | 816 | 784 | 803 | **6,400** | $224 \times 224$ |
| **CPFSK** | 776 | 780 | 787 | 820 | 828 | 792 | 829 | 788 | **6,400** | $224 \times 224$ |
| **DSB-AM** | 813 | 755 | 804 | 785 | 822 | 802 | 805 | 814 | **6,400** | $224 \times 224$ |
| **GFSK** | 789 | 834 | 792 | 777 | 816 | 794 | 800 | 798 | **6,400** | $224 \times 224$ |
| **LFM** | 771 | 810 | 768 | 815 | 773 | 819 | 808 | 836 | **6,400** | $224 \times 224$ |
| **PAM4** | 794 | 822 | 815 | 782 | 802 | 783 | 831 | 771 | **6,400** | $224 \times 224$ |
| **QPSK** | 799 | 802 | 811 | 757 | 837 | 854 | 733 | 807 | **6,400** | $224 \times 224$ |
| **Rect** | 787 | 794 | 810 | 802 | 814 | 777 | 829 | 787 | **6,400** | $224 \times 224$ |
| **StepFM** | 789 | 818 | 744 | 837 | 803 | 804 | 782 | 823 | **6,400** | $224 \times 224$ |
| **Total** | **9,580** | **9,542** | **9,481** | **9,700** | **9,733** | **9,611** | **9,532** | **9,624** | **76,800** | - |

> **Note:** The dataset covers a wide range of noise conditions, from **snr01** (highest noise) to **snr08** (cleanest signal), ensuring model robustness.

## Models
### Network Structure:
![Model](img/model.png)
> **Note:**: Note: The architecture uses the InvResAttentionBlock (IRB), denoted as IRB (C, S, E, D), where C is the number of output channels and S is the stride (S=2 for downsampling, S=1 to maintain resolution). The E parameter defines the expansion ratio for the hidden layer (hidden_dim = in_channels × E), while D specifies the dilation rate used to expand the receptive field without adding parameters.

## Visualization & Results

### Training Progress
The charts below illustrate the model's convergence over 40 epochs. The curves demonstrate high stability with no signs of overfitting, successfully exceeding the target accuracy requirement of > 90%.

| Loss Curve | Accuracy Curve |
| :---: | :---: |
| ![Loss Curve](loss_curve.png) | ![Accuracy Curve](accuracy_curve.png) |

| F1 Score Curve | Precision & Recall Curves |
| :---: | :---: |
| ![F1 Curve](f1_curve.png) | ![Precision Curve](precision_curve.png) <br> ![Recall Curve](recall_curve.png) |

### Confusion Matrices
Confusion matrices provide a detailed analysis of classification performance across all 12 signal types for both Training and Validation sets.

* **Observations:** Signal classes such as `Barker`, `LFM`, and `Rect` achieve near-perfect recognition accuracy. Minor confusion is observed between `16-QAM` and `QPSK` in high-noise environments (low SNR) due to their similar spectrogram characteristics.

| Training Set | Validation Set |
| :---: | :---: |
| ![Train Confusion Matrix](train_confusion_matrix.png) | ![Val Confusion Matrix](val_confusion_matrix.png) |

### Technical Requirements Compliance
The project strictly adheres to the mandatory technical constraints:
* **Trainable Parameters:** The total number of model weights is kept below 100,000.
* **Classification Accuracy:** Final validation accuracy exceeds the 90% threshold.
* **Reproducibility:** A fixed random seed was implemented across all libraries to ensure consistent results.
