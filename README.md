# DeepCIFAR-ImageClassifier

## Overview

This project focuses on building, training, and improving a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes such as airplanes, cars, birds, and more.

The main goal is to develop a baseline CNN model and apply various modifications to enhance its accuracy and generalization capabilities.

---

## Project Structure

- **Data Loading & Preprocessing:**  
  Loading CIFAR-10 dataset via Keras, normalizing pixel values to [0, 1], and one-hot encoding labels.

- **Model Development:**  
  Starting with a baseline CNN architecture and applying enhancements such as dropout, batch normalization, deeper layers, data augmentation, and optimized learning rate scheduling.

- **Training & Evaluation:**  
  Training models, evaluating on test data, plotting training curves, and comparing model performances.

- **Analysis & Visualization:**  
  Including accuracy/loss curves, confusion matrices, and sample predictions to understand model behavior.

---

## Baseline Model Details

- Architecture:  
  - 2 convolutional layers  
  - Max pooling layers  
  - Fully connected dense layers  
  - ReLU activations and softmax output  

- Performance:  
  - Training Accuracy: 77.35%  
  - Validation Accuracy: 68.58%  
  - Test Accuracy: 67.99%  

- Observation:  
  Overfitting was observed with training accuracy much higher than validation accuracy.

---

## Model Improvements and Results

| Model                          | Test Accuracy | Notes                                              |
| ------------------------------|---------------|---------------------------------------------------|
| Baseline Model                 | 67.99%        | Simple CNN with overfitting                        |
| Model with Dropout             | 67.31%        | Reduced overfitting, slightly lower accuracy      |
| Deeper Model + Batch Norm      | 73.64%        | Better accuracy and training stability             |
| Model with Data Augmentation   | 64.39%        | Increased generalization, lower training accuracy  |
| Combined Model (Reg + Aug + BN)| 65.94%        | Mixed effects, better regularization               |
| Best Model (Deep + Dropout + BN + AdamW) | 89.29% | Significantly improved performance with optimized architecture |

---

## Key Techniques That Improved Performance

- **Batch Normalization:**  
  Accelerated training and stabilized gradients, allowing deeper architectures.

- **Dropout:**  
  Controlled overfitting, especially in fully connected layers.

- **Data Augmentation:**  
  Introduced training data variability to enhance generalization.

- **L2 Regularization & Learning Rate Scheduling:**  
  Provided stable convergence and avoided drastic learning rate shifts.

---

## Challenges

- **Overfitting:** Addressed by dropout, batch normalization, and data augmentation.  
- **Training Stability:** Managed via learning rate schedulers like Cosine Decay and ReduceLROnPlateau.  
- **Computational Load:** Deeper models required more training time and resources. Mitigated by mini-batch training and early stopping.  
- **Data Augmentation Impact:** Not all augmentation strategies improved accuracy; tuning required.

---

## Final Conclusion

The project demonstrated that increasing model depth, combined with batch normalization, dropout, and learning rate optimization, significantly improves CIFAR-10 classification accuracy. The best model achieved **89.29% test accuracy**, showing a robust balance of performance and generalization.

Future work could explore transfer learning using pre-trained architectures like ResNet or EfficientNet to push performance further.

---

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/cnn-cifar10-image-classification.git
   cd cnn-cifar10-image-classification

---
## References
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Keras Documentation: https://keras.io
- Batch Normalization Paper: Sergey Ioffe, Christian Szegedy, 2015
- AdamW Optimizer: Loshchilov and Hutter, 2017