"""Türkçe duygu analizi için TensorFlow/Keras eğitim uygulaması."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from data import split_samples


def build_model(train_texts: list[str], max_tokens: int = 3000, sequence_length: int = 40) -> tf.keras.Model:
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_mode="int", output_sequence_length=sequence_length)
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(16))
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,), dtype=tf.string),
            vectorizer,
            tf.keras.layers.Embedding(max_tokens, 48, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def to_arrays(samples: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([text for text, _ in samples]), np.array([label for _, label in samples], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorFlow/Keras ile Türkçe duygu analizi")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/turkce_duygu.keras"))
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    train_samples, validation_samples = split_samples(seed=args.seed)
    train_x, train_y = to_arrays(train_samples)
    validation_x, validation_y = to_arrays(validation_samples)
    model = build_model(train_x.tolist())

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    model.fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    loss, accuracy = model.evaluate(validation_x, validation_y, verbose=0)
    print(f"Doğrulama kaybı: {loss:.4f} | doğruluk: {accuracy:.2%}")

    examples = np.array(["Bu ürünü çok sevdim", "Ne yazık ki hiç çalışmadı"])
    predictions = model.predict(examples, verbose=0).reshape(-1)
    for text, score in zip(examples, predictions):
        label = "olumlu" if score >= 0.5 else "olumsuz"
        print(f"{label:8s} ({score:.3f}) → {text}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Model kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
