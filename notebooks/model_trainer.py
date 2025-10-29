import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import tensorflow as tf
from tensorflow.keras import layers, models, applications
import os, math
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

AUTOTUNE = tf.data.AUTOTUNE

# ---------------- Dataset Loader & Preprocessing ----------------
def load_and_prepare_datasets(train_dir, val_dir, test_dir, image_size=(224,224), batch_size=16, augment=True, seed=42):
    tf.random.set_seed(seed)

    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_names)

    class_counts = [len([f for f in os.listdir(os.path.join(train_dir, cls)) if f.lower().endswith(('.png','.jpg','.jpeg'))]) 
                    for cls in class_names]
    total_train = sum(class_counts)
    print("Detected classes:", class_names)
    print("Train class counts:", dict(zip(class_names, class_counts)))

    steps_per_epoch = math.ceil(total_train / batch_size)
    print("Steps per epoch:", steps_per_epoch)

    train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, labels="inferred", label_mode="int",
        image_size=image_size, batch_size=1, shuffle=True, seed=seed
    ).unbatch()

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir, labels="inferred", label_mode="int",
        image_size=image_size, batch_size=batch_size, shuffle=False
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir, labels="inferred", label_mode="int",
        image_size=image_size, batch_size=batch_size, shuffle=False
    )

    preprocess_fn = tf.keras.applications.vgg19.preprocess_input
    augmentation_layers = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.08, 0.08),
    ], name="augment")

    def preprocess_only(image, label):
        image = tf.cast(image, tf.float32)
        return preprocess_fn(image), label

    def augment_then_preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = augmentation_layers(image, training=True)
        return preprocess_fn(image), label

    per_class_ds = []
    for class_idx in range(num_classes):
        ds_i = train_ds_raw.filter(lambda img, lbl, ci=class_idx: tf.equal(lbl, ci))
        ds_i = ds_i.map(augment_then_preprocess if augment else preprocess_only, num_parallel_calls=AUTOTUNE)
        ds_i = ds_i.shuffle(1000, seed=seed).repeat()
        per_class_ds.append(ds_i)

    train_ds = tf.data.experimental.sample_from_datasets(per_class_ds, weights=[1.0/num_classes]*num_classes, seed=seed)
    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
    print("Training dataset prepared with balanced classes in each batch.")

    val_ds = val_ds.map(preprocess_only, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(preprocess_only, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names, steps_per_epoch

# ---------------- Build VGG19 Model ----------------
def build_vgg19_model(num_classes, input_shape=(224,224,3), dropout_rate=0.5):
    base_model = applications.VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model 

# ---------------- Callbacks ----------------
def get_early_stopping(patience=5):
    return tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

def get_reduce_lr(factor=0.5, patience=3, min_lr=1e-6):
    return tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, min_lr=min_lr)

# ---------------- Train Model ----------------
def train_model(model, train_ds, val_ds, steps_per_epoch, epochs=20):
    callbacks = [get_early_stopping(), get_reduce_lr()]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )
    return history

# ---------------- Evaluation Metrics ----------------
def evaluate_model(model, test_ds, class_names):
    print("\nEvaluating model on test data...")
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("\n✅ Model Evaluation Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # Detailed classification report
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return acc, precision, recall, f1

# ---------------- Usage ----------------
train_dir = r"E:\AI and ML\AI and ML projects\COVID_19_Chest_X_ray_Classifier\Artifacts\10_13_2025_14_38\data_transformation\transformed\train"
val_dir   = r"E:\AI and ML\AI and ML projects\COVID_19_Chest_X_ray_Classifier\Artifacts\10_13_2025_14_38\data_transformation\transformed\validation"
test_dir  = r"E:\AI and ML\AI and ML projects\COVID_19_Chest_X_ray_Classifier\Artifacts\10_13_2025_14_38\data_transformation\transformed\test"

batch_size = 16
image_size = (224,224)

train_ds, val_ds, test_ds, class_names, steps_per_epoch = load_and_prepare_datasets(
    train_dir, val_dir, test_dir, image_size=image_size, batch_size=batch_size
)

model = build_vgg19_model(num_classes=len(class_names), input_shape=(224,224,3))
history = train_model(model, train_ds, val_ds, steps_per_epoch, epochs=1)

# ---------------- Evaluate & Save Model ----------------
evaluate_model(model, test_ds, class_names)

model.save("vgg19_final_model.h5")
print("Model saved as vgg19_final_model.h5")

model.save("vgg19_final_model_tf", save_format="tf")
print("Model saved as TensorFlow SavedModel in folder 'vgg19_final_model_tf'")
