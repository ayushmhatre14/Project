# train.py
import argparse
import os
import tensorflow as tf
from utils import get_image_datasets, plot_history

def train_custom(args):
    from models.custom_cnn import build_simple_cnn
    train_ds, val_ds = get_image_datasets(args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size)
    num_classes = len(train_ds.class_names)
    model = build_simple_cnn(input_shape=(args.img_size, args.img_size, 3), num_classes=num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output, save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    plot_history(history)
    return model

def train_transfer(args):
    from models.transfer_mobilenet import build_mobilenet_head
    train_ds, val_ds = get_image_datasets(args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size)
    num_classes = len(train_ds.class_names)
    model, base = build_mobilenet_head(input_shape=(args.img_size, args.img_size, 3), num_classes=num_classes, base_trainable=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output, save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    ]
    print("Training head...")
    history_head = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head, callbacks=callbacks)
    plot_history(history_head)

    # Fine tune
    base.trainable = True
    fine_tune_at = args.fine_tune_at
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Fine-tuning...")
    history_fine = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_fine, callbacks=callbacks)
    plot_history(history_fine)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/processed', help='Dataset folder or root with class subfolders')
    parser.add_argument('--model', type=str, choices=['custom','transfer'], default='transfer')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--output', type=str, default='best_model.h5')
    parser.add_argument('--epochs_head', type=int, default=8)
    parser.add_argument('--epochs_fine', type=int, default=10)
    parser.add_argument('--fine_tune_at', type=int, default=100)
    args = parser.parse_args()

    if args.model == 'custom':
        train_custom(args)
    else:
        train_transfer(args)

if __name__ == "__main__":
    main()
