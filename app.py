import streamlit as st
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.models import load_model


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered"
)


# =========================================================
# Load Model and Face Detector
# =========================================================

@st.cache_resource
def load_resources():
    model = load_model("face_mask_detector.h5")
    detector = MTCNN()

    return model, detector


model, detector = load_resources()


# =========================================================
# Page Title
# =========================================================

st.title("😷 Face Mask Detector")

st.write(
    "Upload an image and the model will detect "
    "whether the person is wearing a mask."
)


# =========================================================
# Image Upload
# =========================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# Prediction Function
# =========================================================

def predict_mask(img, model, detector):

    # Convert PIL image to NumPy array
    img_array = np.array(img)

    # Detect faces
    results = detector.detect_faces(img_array)

    # If no face is detected
    if len(results) == 0:
        return img_array, []

    results_output = []

    # Process every detected face
    for result in results:

        # Get bounding box
        x, y, width, height = result["box"]

        # Prevent negative coordinates
        x = max(0, x)
        y = max(0, y)

        # Calculate bottom-right coordinates
        x2 = x + width
        y2 = y + height

        # Crop face
        face = img_array[y:y2, x:x2]

        # Skip invalid face
        if face.size == 0:
            continue

        # Resize face
        face = cv2.resize(face, (224, 224))

        # Normalize pixels
        face = face.astype("float32") / 255.0

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Make prediction
        prediction = model.predict(
            face,
            verbose=0
        )[0][0]

        # Determine class
        if prediction >= 0.5:

            label = "With Mask"
            confidence = prediction

            # Green - RGB
            color = (0, 255, 0)

        else:

            label = "Without Mask"
            confidence = 1 - prediction

            # Red - RGB
            color = (255, 0, 0)

        # Store prediction
        results_output.append({
            "box": (x, y, x2, y2),
            "label": label,
            "confidence": confidence,
            "color": color
        })

    # Copy image for drawing
    output = img_array.copy()

    # Draw results
    for result in results_output:

        x, y, x2, y2 = result["box"]

        label = result["label"]

        confidence = result["confidence"]

        color = result["color"]

        # Label text
        text = f"{label} ({confidence * 100:.1f}%)"

        # Draw bounding box
        cv2.rectangle(
            output,
            (x, y),
            (x2, y2),
            color,
            3
        )

        # Label background
        label_y1 = max(0, y - 35)

        cv2.rectangle(
            output,
            (x, label_y1),
            (x2, y),
            color,
            -1
        )

        # Label text
        cv2.putText(
            output,
            text,
            (x + 5, max(22, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    return output, results_output


# =========================================================
# Run Prediction
# =========================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # -----------------------------------------
    # Original Image
    # -----------------------------------------

    st.subheader("Original Image")

    st.image(image)


    # -----------------------------------------
    # Detect Button
    # -----------------------------------------

    if st.button("🔍 Detect Mask"):

        with st.spinner("Detecting face and mask..."):

            output_image, results = predict_mask(
                image,
                model,
                detector
            )


        # -----------------------------------------
        # No Face Detected
        # -----------------------------------------

        if len(results) == 0:

            st.error("❌ No face detected.")


        # -----------------------------------------
        # Face Detected
        # -----------------------------------------

        else:

            st.subheader("Detection Result")

            # Display image with bounding boxes
            st.image(output_image)


            # Display prediction for each face
            for i, result in enumerate(results):

                if result["label"] == "With Mask":

                    st.success(
                        f"🟢 Person {i + 1}: "
                        f"With Mask — "
                        f"{result['confidence'] * 100:.1f}%"
                    )

                else:

                    st.error(
                        f"🔴 Person {i + 1}: "
                        f"Without Mask — "
                        f"{result['confidence'] * 100:.1f}%"
                    )