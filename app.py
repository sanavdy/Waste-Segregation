import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tempfile
import os

model = tf.keras.models.load_model("model.h5")

def predict_func(img):
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    
    n=[]
    s=80

    img = cv2.resize(img, (80, 80))
    
    # Convert image to float32 and normalize
    img = img.astype(np.float32) / 255.0
    
    # Expand dimensions to match the input shape expected by the model
    img = np.expand_dims(img, axis=0)

    img = np.asarray(img)

    # print(img)
    
    # Predict
    result = model.predict(img)
    result=np.where(result >= 0.35, 1, 0)
    print(result)
    if result == 1:
        print("\033[94m" + "This image -> Recyclable" + "\033[0m")
        return("The waste is recyclable")
    elif result == 0:
        print("\033[94m" + "This image -> Organic" + "\033[0m")
        return("The waste is organic")

st.title("WASTE CLASSIFICATION")
st.write("This Waste Classification ML Model employs deep learning techniques to categorize waste types accurately. It preprocesses data, trains on labeled datasets, and evaluates its performance. Once deployed, it integrates with existing systems for real-time or batch processing. Continuous refinement ensures its effectiveness in waste management, aiding in sorting, recycling, and disposal efforts while minimizing environmental impact.")
st.divider()
st.header("Upload an image file")
try:
    
  file = st.file_uploader("Upload File", type=["png","jpg","jpeg"])
  if file:
          temp_dir = tempfile.mkdtemp()
          path = os.path.join(temp_dir, file.name)
          with open(path, "wb") as f:
                  f.write(file.getvalue())

  img = cv2.imread(path)
  result = predict_func(img)

  print(result)

  st.divider()
  st.image(file)
  st.write(result)
except:
    print("Waiting for image")

