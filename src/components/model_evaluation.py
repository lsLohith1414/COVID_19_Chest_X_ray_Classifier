import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ✅ Use non-GUI backend (prevents Tkinter "main thread" errors)
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from tensorflow.keras.models import load_model

from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entities.model_evaluation_config import ModelEvaluationConfig
from src.entities.artifacts_entities import ModelTrainerArtifact, ModelEvaluationArtifact


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CustomException(e)

    def evaluate_model(self, model_path, test_ds, class_names):
        try:
            # 1️⃣ Load the saved model
            logging.info(f"🔍 Loading trained model from: {model_path}")
            model = load_model(model_path)

            # 2️⃣ Create evaluation artifacts directory
            os.makedirs(self.model_evaluation_config.evaluation_dir, exist_ok=True)

            # 3️⃣ Get true & predicted labels
            y_true = np.concatenate([y for _, y in test_ds], axis=0)
            y_pred_probs = model.predict(test_ds)
            y_pred = np.argmax(y_pred_probs, axis=1)

            # 4️⃣ Compute metrics
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')

            logging.info("Model Evaluation Metrics:")
            logging.info(f"Accuracy : {acc:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall   : {recall:.4f}")
            logging.info(f"F1-score : {f1:.4f}")

            # 5️⃣ Save metrics to JSON file
            cls_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            metrics_path = self.model_evaluation_config.matrics_file_name

            with open(metrics_path, "w") as f:
                json.dump({
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "classification_report": cls_report
                }, f, indent=4)

            logging.info(f"Metrics saved at: {metrics_path}")

            # 6️⃣ Confusion Matrix plot
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")

            confusion_matrix_path = self.model_evaluation_config.confusion_matrix_file_name
            plt.savefig(confusion_matrix_path)
            plt.close()
            logging.info(f"Confusion Matrix saved at: {confusion_matrix_path}")

            # ✅ Return metrics for pipeline
            return acc, precision, recall, f1

        except Exception as e:
            raise CustomException(e)

    def initiate_evaluation(self):
        try:
            logging.info("Starting model evaluation process...")

            acc, precision, recall, f1 = self.evaluate_model(
                model_path=self.model_trainer_artifact.trained_model_file_path,
                test_ds=self.model_trainer_artifact.test_ds,
                class_names=self.model_trainer_artifact.class_names
            )

            model_evaluation_artifact = ModelEvaluationArtifact(
                acc=acc,
                precision=precision,
                recall=recall,
                f1=f1
            )

            logging.info("Model evaluation completed successfully.")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e)
