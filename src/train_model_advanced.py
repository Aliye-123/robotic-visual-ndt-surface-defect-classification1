import shutil
import random
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


ORIGINAL_DATASET = Path("dataset/dataset_endustriyel_kare")
SPLIT_DATASET = Path("dataset/dataset_7_split")

MODEL_PATH = "models/model_7sinif_best.keras"
CLASS_NAMES_PATH = "models/class_names_7sinif.json"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp"}

USED_CLASSES = [
    "boyali_hasarsiz",
    "boyali_orta_hasarli",
    "boyali_yuksek_hasarli",
    "boyasiz_hasarsiz",
    "boyasiz_orta_hasarli",
    "boyasiz_yuksek_hasarli",
    "unknown"
]

TRAIN_COUNT = 35
VAL_COUNT = 7
TEST_COUNT = 8

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS_STAGE1 = 18
EPOCHS_STAGE2 = 12
SEED = 42

MODEL_PATH = r"C:\Users\aliye\Desktop\robotik\model_7sinif_best.keras"
CLASS_NAMES_PATH = r"C:\Users\aliye\Desktop\robotik\class_names_7sinif.json"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


if not ORIGINAL_DATASET.exists():
    raise FileNotFoundError(f"Dataset bulunamadi: {ORIGINAL_DATASET}")

if SPLIT_DATASET.exists():
    shutil.rmtree(SPLIT_DATASET)

print("=== 7 SINIF SPLIT BASLADI ===")
for class_name in USED_CLASSES:
    class_dir = ORIGINAL_DATASET / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Sinif klasoru bulunamadi: {class_dir}")

    images = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]

    needed = TRAIN_COUNT + VAL_COUNT + TEST_COUNT
    if len(images) < needed:
        raise ValueError(f"{class_name} icin veri yetersiz. Gerekli={needed}, bulunan={len(images)}")

    images = sorted(images)
    random.shuffle(images)

    train_imgs = images[:TRAIN_COUNT]
    val_imgs = images[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
    test_imgs = images[TRAIN_COUNT + VAL_COUNT:TRAIN_COUNT + VAL_COUNT + TEST_COUNT]

    for split_name, split_imgs in [
        ("train", train_imgs),
        ("val", val_imgs),
        ("test", test_imgs),
    ]:
        target_dir = SPLIT_DATASET / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for img in split_imgs:
            shutil.copy2(img, target_dir / img.name)

    print(f"{class_name}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

print("\n=== SPLIT TAMAMLANDI ===")
print(SPLIT_DATASET)


TRAIN_DIR = SPLIT_DATASET / "train"
VAL_DIR = SPLIT_DATASET / "val"
TEST_DIR = SPLIT_DATASET / "test"

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\n=== SINIFLAR ===")
for i, c in enumerate(class_names):
    print(i, c)

with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(300).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.08),
    keras.layers.RandomZoom(0.10),
    keras.layers.RandomContrast(0.10),
    keras.layers.RandomTranslation(0.05, 0.05),
])

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.35)(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

print("\n=== 1. ASAMA EGITIM ===")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks
)

print("\n=== 2. ASAMA FINE-TUNE ===")
base_model.trainable = True

# son 40 katman acik kalsin
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=8e-6),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1 + EPOCHS_STAGE2,
    initial_epoch=len(history1.history["accuracy"]),
    callbacks=callbacks
)

model = keras.models.load_model(MODEL_PATH)

test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

y_true = np.concatenate([y for _, y in test_ds], axis=0)
y_prob = model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.title("Confusion Matrix - 7 Sinif (Gelismis)")
plt.tight_layout()
plt.show()

acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss = history1.history["loss"] + history2.history["loss"]
val_loss = history1.history["val_loss"] + history2.history["val_loss"]

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, acc, label="train_acc")
plt.plot(epochs_range, val_acc, label="val_acc")
plt.title("Accuracy - 7 Sinif (Gelismis)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, loss, label="train_loss")
plt.plot(epochs_range, val_loss, label="val_loss")
plt.title("Loss - 7 Sinif (Gelismis)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nModel kaydedildi: {MODEL_PATH}")
print(f"Sinif isimleri kaydedildi: {CLASS_NAMES_PATH}")
