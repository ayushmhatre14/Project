# train.py
import os
import json
import argparse
import tensorflow as tf
from tensorflow import keras
import keras.layers as layers   # ✅ Updated import
import numpy as np

# Reduce noisy TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def build_model(input_shape=(224,224,3), num_classes=2):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.1)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="simple_cnn")
    return model

def main(args):
    data_dir = args.data_dir
    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size
    epochs = args.epochs

    # ❌ If folder is missing → throw clear error
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"DATA_DIR '{data_dir}' not found. Put dataset here or change --data-dir."
        )

    # ✅ Load dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Class names:", class_names)

    # Prefetch for speed
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build + compile model
    model = build_model(input_shape=img_size + (3,), num_classes=num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_accuracy"),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # ===========================================
    # NEW ✔ FIX: Save model with valid extension
    # ===========================================
    model.save("saved_model.keras")   # REQUIRED FIX
    print("Model saved as 'saved_model.keras'")

    # Save class names
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    print("✅ Training finished. Model saved to 'saved_model.keras' and 'best_model.h5'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/PlantVillage",
                        help="Path to dataset directory (should have subfolders per class).")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    main(args)
