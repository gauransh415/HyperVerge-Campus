# Multi-Class Image Classifier using ResNet-18

This project implements a multi-class image classifier using a pre-trained ResNet-18 model. The model is fine-tuned on a labeled dataset of animal images and generates predictions for an unlabeled test set. The entire workflow, from data exploration to model inference, is contained within the `main.ipynb` Jupyter Notebook.

---

## ✨ Features

* **Model:** Utilizes a **ResNet-18** model pre-trained on the ImageNet dataset.
* **Framework:** Built with **PyTorch**.
* **Data Handling:** Includes comprehensive data exploration, visualization, and preprocessing steps.
* **Data Augmentation:** Applies transformations like resizing, random flips, rotations, and color jittering to enhance model robustness.
* **Training:** Features a complete training loop with a validation phase, Adam optimizer, and a learning rate scheduler.
* **Inference:** Generates and saves predictions for a given test dataset into a `.csv` file.

---

## 📂 Project Structure

The project is organized into the following directory structure:
Of course. Here is the complete README.md content in a single markdown block for you to copy.
Markdown

# Multi-Class Image Classifier using ResNet-18

This project implements a multi-class image classifier using a pre-trained ResNet-18 model. The model is fine-tuned on a labeled dataset of animal images and generates predictions for an unlabeled test set. The entire workflow, from data exploration to model inference, is contained within the `main.ipynb` Jupyter Notebook.

---

## ✨ Features

* **Model:** Utilizes a **ResNet-18** model pre-trained on the ImageNet dataset.
* **Framework:** Built with **PyTorch**.
* **Data Handling:** Includes comprehensive data exploration, visualization, and preprocessing steps.
* **Data Augmentation:** Applies transformations like resizing, random flips, rotations, and color jittering to enhance model robustness.
* **Training:** Features a complete training loop with a validation phase, Adam optimizer, and a learning rate scheduler.
* **Inference:** Generates and saves predictions for a given test dataset into a `.csv` file.

---

## 📂 Project Structure

The project is organized into the following directory structure:

```multiclass-image-classifier/
├── README.md                # Project documentation (this file)
├── main.ipynb               # Jupyter Notebook with the complete workflow
└── test_predictions.csv     # Output file with predictions for the test set
```

## 🚀 Workflow

The project follows a standard machine learning pipeline for image classification.

### 1. Data Loading and Preprocessing

* The labeled dataset consists of 779 images across 10 classes: `cane`, `cavallo`, `elefante`, `farfalla`, `gallina`, `gatto`, `mucca`, `pecora`, `ragno`, and `scoiattolo`.
* A custom `ImageDataset` class is used to load images and their corresponding labels from a CSV file.
* The data is split into an **80% training set** and a **20% validation set**.
* **Data Augmentation and Transformation:**
    * **Training Data:** Images are resized to 224x224 pixels and undergo random horizontal flips, rotations, color jittering, and affine transformations to prevent overfitting.
    * **Validation & Test Data:** Images are simply resized to 224x224 pixels.
    * All images are normalized using ImageNet's mean and standard deviation.

### 2. Model Architecture


* The classifier is based on the **ResNet-18** architecture, which is known for its efficiency and performance.
* The pre-trained model's final fully connected layer is replaced with a new linear layer that has an output size matching the number of classes in our dataset (10).

### 3. Training

* The model was trained for **25 epochs** using the following configuration:
    * **Loss Function:** Cross-Entropy Loss (`nn.CrossEntropyLoss`)
    * **Optimizer:** Adam (`optim.Adam`)
    * **Initial Learning Rate:** 0.001
    * **Learning Rate Scheduler:** The learning rate is reduced by a factor of 0.1 every 7 epochs (`StepLR`).
    * **Batch Size:** 32
* The model weights that achieved the **best validation accuracy** (approximately **73.72%**) were saved and used for the final inference step.

### 4. Inference

* The trained model is loaded with its best-performing weights.
* An `UnlabeledImageDataset` class processes the test images.
* The model predicts the class for each image in the test set.

---

## ⚙️ Setup and Usage

### Prerequisites

Ensure you have Python installed, along with the necessary libraries. You can install them using pip:

```bash
pip install torch torchvision pandas numpy matplotlib seaborn
