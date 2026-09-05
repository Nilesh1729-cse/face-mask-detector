# 😷 Face Mask Detector

A simple computer-vision application that detects whether a person is wearing a face mask.

## 🚀 Live Demo

**Try the deployed application:**  
https://face-mask-detector-fe5bzywvc6zsrrzyjjueou.streamlit.app/

The application allows users to upload an image and get a mask/no-mask prediction.

---

## 📌 Project Overview

The goal of this project is to build a face-mask detection system that classifies a detected face into:

- 🟢 **With Mask**
- 🔴 **Without Mask**

The project combines two main computer-vision steps:

1. **MTCNN** finds where the face is.
2. **CNN** classifies the detected face as `With Mask` or `Without Mask`.

The trained model is connected to a **Streamlit web application**, so users can interact with the project through a browser.

---

# 👤 How Does It Work?

For a non-technical understanding:

```text
Image
  ↓
Find the face
  ↓
Focus only on the face
  ↓
Analyze visual patterns learned during training
  ↓
Predict:
🟢 With Mask
or
🔴 Without Mask
```

The model does **not** use a manually written rule such as:

> "If the mouth is covered, it must be a mask."

Instead, the CNN learns useful visual patterns from the training examples.

---

# 🔍 Technical Working

## 1. Face Detection with MTCNN

MTCNN stands for **Multi-task Cascaded Convolutional Networks**.

Its job is:

> **"Where is the face in this image?"**

MTCNN returns a bounding box around each detected face.

```text
Full Image
     ↓
    MTCNN
     ↓
┌───────────────┐
│     FACE      │
└───────────────┘
```

Only the detected face is passed to the mask-classification CNN.

---

## 2. Face Preprocessing

The detected face is:

- Cropped
- Resized to **224 × 224 pixels**
- Converted into numerical pixel values
- Normalized from `0–255` to `0–1`

This gives the classifier a consistent input format.

---

## 3. CNN Classification

A custom **Convolutional Neural Network (CNN)** performs the mask classification.

The current architecture is:

```text
Input: 224 × 224 × 3
        ↓
Conv2D (32)
        ↓
MaxPooling
        ↓
Conv2D (64)
        ↓
MaxPooling
        ↓
Conv2D (128)
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense (128)
        ↓
Dropout (0.5)
        ↓
Sigmoid Output
```

The convolution layers learn visual patterns from the training images.

These may include patterns associated with:

- Mask shape
- Mask edges
- Texture
- Color/pixel patterns
- Coverage of the lower face
- Differences between masked and unmasked faces

These features are **learned automatically** rather than being manually programmed.

---

# 🎯 How Is the Final Prediction Made?

The final layer uses a **sigmoid activation function**, producing a value between `0` and `1`.

The project uses a threshold of `0.5`:

```text
Prediction >= 0.5
        ↓
   🟢 With Mask

Prediction < 0.5
        ↓
  🔴 Without Mask
```

For example:

```text
Prediction = 0.93
→ With Mask
```

```text
Prediction = 0.08
→ Without Mask
```

For an `Without Mask` prediction, the displayed score is:

```text
1 - prediction
```

### Important note about "confidence"

The displayed confidence comes from the model's sigmoid output. It should be treated as a **model score**, not a guarantee that the prediction is correct.

---

# 📚 Dataset

The dataset contains:

```text
train/
├── with_mask/
└── without_mask/
```

Dataset summary:

```text
Original images : 1,279
With mask       : 644
Without mask    : 635
```

After face detection using MTCNN:

```text
Successfully detected faces : 1,253
```

The remaining images were skipped because a usable face was not detected.

The processed dataset was split into approximately:

```text
80% → Training
20% → Testing
```

---

# 🧠 What Does the Model Actually Learn?

A human can look at a masked face and immediately understand:

> "This person is wearing a mask."

A CNN works differently.

Conceptually:

```text
Training Images
      ↓
Learn visual patterns
      ↓
Create internal feature representations
      ↓
Process a new face
      ↓
Calculate sigmoid score
      ↓
With Mask / Without Mask
```

Therefore, the model does not truly "understand" a mask like a human. It learns statistical visual patterns from the examples it was trained on.

This explains why a clearly masked person can sometimes still receive an incorrect prediction.

---

# 🌐 Streamlit Web Application

The trained model is deployed as a **Streamlit web application**.

### Web application flow

```text
User
 ↓
Upload Image
 ↓
MTCNN Face Detection
 ↓
Face Crop
 ↓
Resize + Normalize
 ↓
CNN Prediction
 ↓
Display Bounding Box + Result
```

The application displays:

```text
🟢 With Mask
```

or:

```text
🔴 Without Mask
```

along with the model score.

### Live Demo

https://face-mask-detector-fe5bzywvc6zsrrzyjjueou.streamlit.app/

---

# ⚠️ Current Limitations

The current project is a **working prototype**, but it is not yet a production-grade face-mask detector.

## 1. Limited Dataset Size

Only about **1,002 images** are used for training after preprocessing and the train/test split.

A small dataset limits how well a CNN trained from scratch can generalize to unseen situations.

## 2. Generalization Problems

The model may struggle with images that are very different from the training data, such as:

- Unusual mask designs
- Different lighting
- Different cameras
- Different backgrounds
- Side profiles
- Low-quality images
- Partially visible faces

Therefore, a visually obvious masked face can still be classified incorrectly.

## 3. Large Model Compared with Dataset

The current CNN has approximately:

```text
11.17 million trainable parameters
```

The `Flatten()` layer followed by a dense layer creates many parameters.

With relatively little training data, this can increase the risk of **overfitting**.

In simple terms:

> The model may learn the training examples very well without learning a sufficiently general rule for all new images.

## 4. Dependence on MTCNN

The CNN can classify a face only after a face has been detected.

If MTCNN cannot find the face, the system cannot make a mask prediction for that face.

Difficult cases include:

- Very small faces
- Extreme side profiles
- Poor lighting
- Blurry images
- Heavy occlusion
- Unusual poses

## 5. Multiple Faces

The application can process multiple detected faces, but performance can become less reliable when faces are:

- Very small
- Overlapping
- Partially hidden
- Far from the camera

## 6. Real-Time Performance

MTCNN can be computationally expensive when used repeatedly on video frames.

This may reduce FPS on systems without strong hardware.

---

# 🔮 Future Scope

There is significant scope for improving the project.

## 1. Transfer Learning

The most important improvement would be to use a pretrained network such as:

- **MobileNetV2**
- **EfficientNet**
- **ResNet**
- **Xception**

Instead of learning all visual features from scratch, a pretrained network starts with features learned from large image datasets.

A lightweight model such as **MobileNetV2** could provide a good balance between accuracy and speed for this type of application.

## 2. Data Augmentation

Training images can be varied artificially using:

- Rotation
- Horizontal flipping
- Zoom
- Cropping
- Translation
- Brightness changes

This increases the variety seen during training and can help reduce overfitting.

## 3. Larger and More Diverse Dataset

A larger dataset could include:

- Different mask colors
- Different mask materials
- Different people
- Different lighting conditions
- Different camera angles
- Indoor and outdoor environments
- Partial occlusions
- More real-world photographs

This should improve generalization.

## 4. Better Face Detection

MTCNN could potentially be replaced or supplemented with faster or more robust face detectors such as:

- MediaPipe Face Detection
- RetinaFace
- YOLO-based face detectors

This could improve detection quality and real-time performance.

## 5. Better Evaluation

Accuracy alone does not tell the complete story.

Future versions should report:

- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

This is especially useful for understanding false positives and false negatives.

## 6. Prediction Smoothing for Webcam

For a live camera, predictions can sometimes fluctuate:

```text
With Mask
Without Mask
With Mask
Without Mask
```

even when the person is barely moving.

A future version could average several consecutive predictions to make the output more stable.

## 7. Faster Real-Time Detection

The webcam version can be optimized by using:

- A lightweight face detector
- MobileNet-based classification
- Lower webcam resolution
- Frame skipping
- GPU acceleration
- Prediction caching/smoothing

This can increase FPS and reduce CPU usage.

## 8. Improved Web Interface

The Streamlit application can be extended with:

- Image upload
- Browser camera input
- Live detection
- Detection history
- Confidence charts
- Multiple-person statistics
- Better visual reporting

---

# 🛠️ Technologies Used

```text
Python
TensorFlow
Keras
MTCNN
OpenCV
NumPy
Pillow
Matplotlib
Scikit-learn
Streamlit
Git
GitHub
Git LFS
```

---

# 📁 Project Structure

```text
face-mask-detector/
│
├── app.py
├── face_mask_detector.h5
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
```

### Main files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web application |
| `face_mask_detector.h5` | Trained CNN model |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `.gitignore` | Files excluded from Git |
| `.gitattributes` | Git LFS configuration |

The `.h5` model is stored using **Git LFS** because it is larger than GitHub's standard per-file limit.

---

# ✅ Current Project Status

```text
✅ Dataset preparation
✅ Face detection
✅ Face preprocessing
✅ CNN training
✅ Test evaluation
✅ New-image prediction
✅ Model saving
✅ Streamlit web application
✅ Multiple-face detection
✅ GitHub repository
✅ Git LFS model storage
✅ Streamlit deployment
```

---

# 🎓 Project Takeaway

This project demonstrates a complete machine-learning and computer-vision workflow:

```text
Dataset
   ↓
Data Processing
   ↓
Face Detection
   ↓
CNN Training
   ↓
Evaluation
   ↓
Prediction
   ↓
Web Application
   ↓
Cloud Deployment
```

The key idea is that the system uses **two different stages**:

> **MTCNN answers: "Where is the face?"**

> **CNN answers: "Does this face look more like the masked or unmasked examples it learned from?"**

The current system is a useful prototype that demonstrates the complete journey from a dataset to a deployed machine-learning application.

Its biggest improvement area is **generalization to unseen real-world images**, which can be improved through transfer learning, more diverse data, augmentation, stronger evaluation, and faster detection models.
