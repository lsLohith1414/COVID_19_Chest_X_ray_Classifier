import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import os, math
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.logging.logger import logging
from src.exception.exception import CustomException   

from src.entities.model_training_config import ModelTrainingConfig
from src.entities.artifacts_entities import ModelTrainerArtifact, DataTransformationArtifact
from src.constants import model_training_constants



class ModelTrainer:
    def __init__(self, model_training_config:ModelTrainingConfig, model_transformation_artifact:DataTransformationArtifact):

        try:
            self.model_training_config = model_training_config
            self.model_transformation_artifact = model_transformation_artifact

        except Exception as e:
            raise CustomException(e)
        


    
        # ---------------- Dataset Loader & Preprocessing ----------------
    def load_and_prepare_datasets(self, train_dir, val_dir, test_dir, image_size, batch_size, augment, seed):
        # ✅ Set random seed for reproducibility (same shuffle order, augmentations, etc.)

        try:
            tf.random.set_seed(seed)
            AUTOTUNE = tf.data.AUTOTUNE
        
            # 🧭 1️⃣ Detect classes and gather stats

            # ✅ Get all subfolder names under train_dir → each represents one class
            class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
            num_classes = len(class_names)

            # ✅ Count number of images in each class folder (for info and balancing)
            class_counts = [len([f for f in os.listdir(os.path.join(train_dir, cls))
                                if f.lower().endswith(('.png','.jpg','.jpeg'))]) 
                            for cls in class_names]
            total_train = sum(class_counts)

            # ✅ Display what classes were found and how many images per class
            print("Detected classes:", class_names)
            print("Train class counts:", dict(zip(class_names, class_counts)))

            # ✅ Compute total training steps per epoch (for model.fit)
            steps_per_epoch = math.ceil(total_train / batch_size)
            print("Steps per epoch:", steps_per_epoch)


            # 🧱 2️⃣ Load training dataset (unbatched for per-class processing)

            # ✅ Load all training images from folders.
            # label_mode="int" → numeric labels (0,1,2)
            # batch_size=1 → each batch = single image, because we will unbatch() anyway
            train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
                train_dir, 
                labels="inferred", 
                label_mode="int",
                image_size=image_size, 
                batch_size=1, 
                shuffle=True, 
                seed=seed
            ).unbatch()  # ✅ unbatch() flattens so each element = (image, label)


        
            # 🧾 3️⃣ Load validation and test datasets (normal, no unbatch)

            # ✅ Validation dataset (no shuffle, no unbatch)
            val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                val_dir, 
                labels="inferred", 
                label_mode="int",
                image_size=image_size, 
                batch_size=batch_size, 
                shuffle=False
            )

            # ✅ Test dataset (same)
            test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                test_dir, 
                labels="inferred", 
                label_mode="int",
                image_size=image_size, 
                batch_size=batch_size, 
                shuffle=False
            )



            # 🎨 4️⃣ Define preprocessing & augmentation

                # ✅ Preprocessing function specific to VGG19 — scales pixels from [0,255] to VGG19 expected range
            preprocess_fn = tf.keras.applications.vgg19.preprocess_input

            # ✅ Data augmentation pipeline (applied only during training)
            augmentation_layers = tf.keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomZoom(0.08, 0.08),
            ], name="augment")


            # 🧩 5️⃣ Define helper functions for preprocessing
                # ✅ Only preprocessing (used for val/test)
            def preprocess_only(image, label):
                image = tf.cast(image, tf.float32)
                return preprocess_fn(image), label

            # ✅ Augmentation + preprocessing (used for training)
            def augment_then_preprocess(image, label):
                image = tf.cast(image, tf.float32)
                image = augmentation_layers(image, training=True)
                return preprocess_fn(image), label




            # ⚖️ 6️⃣ Balance dataset across classes

            per_class_ds = []
            for class_idx in range(num_classes):
                # ✅ Select only samples from a specific class
                ds_i = train_ds_raw.filter(lambda img, lbl, ci=class_idx: tf.equal(lbl, ci))
                
                # ✅ Apply augmentation (if enabled) and preprocessing to this class
                ds_i = ds_i.map(augment_then_preprocess if augment else preprocess_only,num_parallel_calls=AUTOTUNE)
                
                # ✅ Shuffle within class and repeat infinitely (for continuous sampling)
                ds_i = ds_i.shuffle(1000, seed=seed).repeat()
                
                # ✅ Add this per-class dataset to the list
                per_class_ds.append(ds_i)


            
            # 🔀 7️⃣ Combine per-class datasets (balanced sampling)

            # ✅ Combine all class datasets so batches are class-balanced
            train_ds = tf.data.experimental.sample_from_datasets(
                per_class_ds,                         # list of per-class datasets
                weights=[1.0/num_classes]*num_classes, # equal sampling probability per class
                seed=seed
            )


            # 📦 8️⃣ Final batching and performance optimization

            # ✅ Batch the dataset and prefetch next batch for speed
            train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
            print("Training dataset prepared with balanced classes in each batch.")

            # ✅ Validation and test datasets: only preprocess + prefetch
            val_ds = val_ds.map(preprocess_only, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
            test_ds = test_ds.map(preprocess_only, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)



            # 🔚 9️⃣ Return everything

            # ✅ Return processed datasets and useful info
            return train_ds, val_ds, test_ds, class_names, steps_per_epoch, num_classes
        
        except Exception as e:
            raise CustomException(e)
    




    # ---------------- Build VGG19 Model ----------------
    def build_vgg19_model(self, num_classes, input_shape, dropout_rate):

        try:

            base_model = applications.VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
            base_model.trainable = False
            x = layers.GlobalAveragePooling2D()(base_model.output)
            x = layers.Dense(512, activation="relu")(x)
            x = layers.Dropout(dropout_rate)(x)
            outputs = layers.Dense(num_classes, activation="softmax")(x)
            model = models.Model(inputs=base_model.input, outputs=outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss=model_training_constants.LOSS_FUNCTION,
                metrics=model_training_constants.METRICS
            )
            return model 
        
        except Exception as e:
            raise CustomException(e)
    


    # ---------------- Callbacks ----------------
    def get_early_stopping(self,patience=5):
        return tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    def get_reduce_lr(self,factor=0.5, patience=3, min_lr=1e-6):
        return tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, min_lr=min_lr)

    # ---------------- Train Model ----------------
    def train_model(self,model, train_ds, val_ds, steps_per_epoch, epochs):

        try:
            callbacks = [self.get_early_stopping(), self.get_reduce_lr()]
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks
            )
            return history
        
        except Exception as e:
            raise CustomException(e)
        

    



       
    
    def initiate_model_training(self):

        try:

            logging.info("Enter Model Training Component")

            logging.info("Start loading the data from artifacts and convers in to tensor data")
            train_ds, validate_ds, test_ds, class_names, steps_per_epoch,num_classes  = self.load_and_prepare_datasets(train_dir=self.model_transformation_artifact.transformed_train_file_path,
                                        test_dir=self.model_transformation_artifact.transformed_test_file_path,
                                        val_dir=self.model_transformation_artifact.transformed_validate_file_path,
                                        image_size= model_training_constants.IMAGE_SIZE,
                                        batch_size=model_training_constants.BATCH_SIZE,
                                        seed= model_training_constants.RANDOM_SEED,
                                        augment=model_training_constants.AUGMENT
                                        )
            logging.info("Data loaded succussfully")
            

            
            model = self.build_vgg19_model(num_classes=num_classes,
                                        input_shape= model_training_constants.INPUT_SHAPE,
                                        dropout_rate= model_training_constants.DROPOUT_RATE
                                        )
            
            logging.info("Model training is started")

            history = self.train_model(model=model,
                                    train_ds=train_ds,
                                    val_ds=validate_ds,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs= model_training_constants.EPOCHS
                                    )
            
            logging.info("Model trained successfully")
            
            # ---------------- & Save Model ----------------

            trained_model_file_path = self.model_training_config.trained_model_file_path
            trained_model_path = os.path.dirname(trained_model_file_path)
            os.makedirs(trained_model_path, exist_ok=True)
    
            model.save(self.model_training_config.trained_model_file_path)
            # New native Keras format (recommended)


            # ✅ Also save a copy in final_models folder
            final_model_dir = os.path.join("final_models")
            os.makedirs(final_model_dir, exist_ok=True)
            final_model_path = os.path.join(final_model_dir, model_training_constants.MODEL_FILE_NAME)
            model.save(final_model_path)

            model_training_artifacts = ModelTrainerArtifact(trained_model_file_path=trained_model_file_path,
                                                            test_ds=test_ds,
                                                            class_names=class_names
                                                            )

            logging.info(f"Model saved in this path: {trained_model_file_path}")

            return model_training_artifacts


        except Exception as e:
            raise CustomException(e)

    