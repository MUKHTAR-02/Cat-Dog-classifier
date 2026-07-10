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

    # Check if folder exists
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' not found.")
        continue

    label = categories.index(category)

    for image_name in os.listdir(folder):

        image_path = os.path.join(folder, image_name)

        image = cv2.imread(image_path)

        # Skip invalid files
        if image is None:
            continue

        # Resize image
        image = cv2.resize(image, (64, 64))

        # Flatten image into a 1D vector
        image = image.flatten()

        data.append(image)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

print("Total Images :", len(data))

# Stop if dataset is empty
if len(data) == 0:
    print("No images found in dataset.")
    exit()

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

model = LogisticRegression(
    max_iter=5000,
    solver="lbfgs"
)

model.fit(X_train, y_train)

print("\nModel Training Complete!")

# -----------------------------
# Step 4: Test Accuracy
# -----------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy : {accuracy * 100:.2f}%")

# -----------------------------
# Step 5: Predict Images from test-imgs Folder
# -----------------------------

test_folder = "test-imgs"

if not os.path.exists(test_folder):
    print(f"\nError: '{test_folder}' folder not found.")
    exit()

print("\nPredictions:\n")

image_found = False

for image_name in os.listdir(test_folder):

    image_path = os.path.join(test_folder, image_name)

    test_image = cv2.imread(image_path)

    # Skip non-image files
    if test_image is None:
        continue

    image_found = True

    test_image = cv2.resize(test_image, (64, 64))
    test_image = test_image.flatten()
    test_image = test_image.reshape(1, -1)

    prediction = model.predict(test_image)

    if prediction[0] == 0:
        print(f"{image_name} --> 🐱 Cat")
    else:
        print(f"{image_name} --> 🐶 Dog")

if not image_found:
    print("No valid images found inside the 'test-imgs' folder.")
