import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Step 1: Load Dataset
# -----------------------------

data = []
labels = []

categories = ["cats", "dogs"]

for category in categories:

    folder = os.path.join("dataset", category)

    label = categories.index(category)

    for image_name in os.listdir(folder):

        image_path = os.path.join(folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize image
        image = cv2.resize(image, (64, 64))

        # Convert image into one long vector
        image = image.flatten()

        data.append(image)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

print("Total Images :", len(data))

# -----------------------------
# Step 2: Split Dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

print("Training Images :", len(X_train))
print("Testing Images :", len(X_test))

# -----------------------------
# Step 3: Train Model
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Training Complete!")

# -----------------------------
# Step 4: Test Accuracy
# -----------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

# -----------------------------
# Step 5: Predict New Image
# -----------------------------

test_image = cv2.imread("test.jpg")

test_image = cv2.resize(test_image, (64, 64))

test_image = test_image.flatten()

test_image = test_image.reshape(1, -1)

prediction = model.predict(test_image)

print("\nPrediction")

if prediction[0] == 0:
    print("🐱 Cat")
else:
    print("🐶 Dog")
