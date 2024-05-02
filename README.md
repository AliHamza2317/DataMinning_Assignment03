DataMinning_Assignment03

Credit Card Fraud Detection using Generative Adversarial Networks (GANs) and Autoencoders

This repository contains the implementation of a fraud detection model using a Generative Adversarial Network (GAN). The model employs an autoencoder as the generator to reconstruct credit card transaction data and a discriminator to differentiate between real and reconstructed transactions. The goal is to identify anomalies in transaction data that may indicate fraudulent activities.

Table of Contents

- Introduction
- Dataset
- Model Details
- Dependencies
- Usage
- Results

Introduction

Credit card fraud is a significant concern for financial institutions and consumers alike. Traditional methods of fraud detection often struggle to keep up with evolving fraud tactics. This project explores the use of advanced machine learning techniques to improve fraud detection accuracy.

Dataset

The model is trained on the "creditcard.csv" file, which should contain typical credit card transactions with a label indicating fraud.



Model Details
Data Preprocessing
    The data is scaled using StandardScaler.
    Missing values are dropped (if any).

Autoencoder
    The autoencoder is used to learn efficient data codings in an unsupervised manner.
    It is a key component of the GAN, acting as the generator.

Discriminator
    The discriminator is a simple neural network that learns to differentiate between real and reconstructed data.
    It uses contrastive loss to enhance its differentiation capabilities.

GAN Integration
    The GAN combines the autoencoder and discriminator.
    It is trained alternately on real and generated data to improve both components.

Dependencies
- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Scikit-learn

Usage

1. Clone the repository:
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   
2. Download the dataset `creditcard.csv` and place it in the `data/` directory.

3. Run the `main.py` script:
   python main.py

   This will train the model and output the Precision-Recall AUC score as well as a classification report.

Results

After training, the model achieves a Precision-Recall AUC score of 0.90. The classification report provides detailed metrics on the model's performance, including precision, recall, and F1-score.
Anomaly detection is crucial in identifying potential cases of credit card fraud. By leveraging GANs and autoencoders, this model demonstrates promising results in detecting fraudulent transactions.

Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.
