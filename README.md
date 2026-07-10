# 🐱🐶 Cat vs Dog Classifier (Machine Learning)

A beginner-friendly **Machine Learning Classification** project that classifies images as **Cat** or **Dog** using **Logistic Regression** and **OpenCV**.

This project is designed for students who are learning the complete Machine Learning workflow from scratch.

---

# 📌 Features

- Load image dataset
- Image preprocessing
- Resize images
- Convert images into feature vectors
- Split dataset into training and testing sets
- Train a Logistic Regression model
- Evaluate model accuracy
- Predict a new image

---

# 🛠 Technologies Used

- Python 3
- NumPy
- OpenCV
- Scikit-learn

---

# 📂 Project Structure

```text
CatDogClassifier/
│
├── dataset/
│   ├── cats/
│   │   ├── cat1.jpg
│   │   ├── cat2.jpg
│   │   └── ...
│   │
│   └── dogs/
│       ├── dog1.jpg
│       ├── dog2.jpg
│       └── ...
│
├── test.jpg
├── cat_dog_classifier.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/CatDogClassifier.git

cd CatDogClassifier
```

Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually

```bash
pip install numpy opencv-python scikit-learn
```

---

# 📁 Dataset

Create the following folder structure.

```text
dataset/

    cats/

        cat1.jpg
        cat2.jpg
        ...

    dogs/

        dog1.jpg
        dog2.jpg
        ...
```

Recommended dataset size:

- 100 Cat Images
- 100 Dog Images

You can also use a larger dataset for better accuracy.

---

# 🚀 Run the Project

```bash
python cat_dog_classifier.py
```

---

# 📊 Machine Learning Pipeline

```
Dataset
   │
   ▼
Load Images
   │
   ▼
Resize Images (64×64)
   │
   ▼
Flatten Images
   │
   ▼
Feature Vector
   │
   ▼
Train-Test Split
   │
   ▼
Logistic Regression
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Prediction
```

---

# 📚 Workflow

### Step 1

Load images from the dataset.

---

### Step 2

Resize every image to **64 × 64** pixels.

---

### Step 3

Flatten each image into a one-dimensional feature vector.

---

### Step 4

Split the dataset into

- 80% Training Data
- 20% Testing Data

---

### Step 5

Train a Logistic Regression model.

---

### Step 6

Evaluate the model using Accuracy.

---

### Step 7

Predict whether a new image is a Cat or Dog.

---

# 📈 Example Output

```text
Total Images : 200

Training Images : 160

Testing Images : 40

Model Training Complete!

Accuracy : 84.50%

Prediction

🐶 Dog
```

---

# 🧠 Concepts Covered

- Artificial Intelligence
- Machine Learning
- Supervised Learning
- Classification
- Dataset
- Features
- Labels
- Image Preprocessing
- Train-Test Split
- Logistic Regression
- Model Training
- Prediction
- Accuracy

---

# ⚠️ Limitations

This project uses **flattened pixel values** as features.

While this approach is excellent for learning Machine Learning concepts, it is **not the best method for image classification**.

For higher accuracy, Deep Learning models such as **Convolutional Neural Networks (CNNs)** are recommended.

---

# 🚀 Future Improvements

- Save the trained model using Joblib
- Build a prediction script without retraining
- Add a graphical user interface (GUI)
- Support drag-and-drop image prediction
- Train using CNN for better accuracy
- Deploy as a web application using Flask or FastAPI

---

# 📄 License

This project is open-source and available under the MIT License.

---

# 👨‍💻 Author

Created for educational purposes to help students understand the complete Machine Learning Classification workflow from scratch.

If you found this project helpful, consider giving it a ⭐ on GitHub.
