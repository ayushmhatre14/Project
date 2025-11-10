# predict.py
import argparse
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import os

def load_and_prep(img_path, img_size):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to saved model (.h5 or SavedModel dir)')
    parser.add_argument('--img', type=str, required=True, help='Image path to predict')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--class_names', type=str, default='', help='Comma-separated class names; otherwise inferred from model training not available')
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    x = load_and_prep(args.img, args.img_size)
    # If model expects preprocess_input (MobileNet), the saved model should contain it; else we can rescale
    try:
        preds = model.predict(x)
    except Exception:
        # try simple rescaling
        x = x / 255.0
        preds = model.predict(x)

    class_names = args.class_names.split(',') if args.class_names else None
    pred_idx = int(np.argmax(preds, axis=1)[0])
    conf = float(np.max(preds))
    if class_names:
        print(f"Predicted: {class_names[pred_idx]} (confidence {conf:.3f})")
    else:
        print(f"Predicted index: {pred_idx} (confidence {conf:.3f})")
