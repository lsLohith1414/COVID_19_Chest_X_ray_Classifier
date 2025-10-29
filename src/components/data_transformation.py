import os
from PIL import Image
import joblib
import numpy as np

from src.exception.exception import CustomException
from src.logging.logger import logging
from src.entities.data_transformation_config import DataTransformationConfig
from src.entities.artifacts_entities import DataIngestionAftifacts
from src.entities.artifacts_entities import DataTransformationArtifact
from src.constants.global_constants import IMAGE_SIZE


class DataTransformation:
    def __init__(self, data_ingestion_artifacts:DataIngestionAftifacts, data_transformation_config:DataTransformationConfig):

        try:
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise CustomException(e)


    def transform_and_save_images(self, source_dir, dest_dir):
        """
        Resize and normalize images and save as normal image files in the destination folder
        """
        try:
            if not os.path.exists(source_dir):
                raise FileNotFoundError(f"Source directory not found: {source_dir}")

            for class_name in os.listdir(source_dir):
                class_source_path = os.path.join(source_dir, class_name)
                if not os.path.isdir(class_source_path):
                    continue

                class_dest_path = os.path.join(dest_dir, class_name)
                os.makedirs(class_dest_path, exist_ok=True)

                for img_name in os.listdir(class_source_path):
                    img_path = os.path.join(class_source_path, img_name)
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(IMAGE_SIZE)

                    # Normalize pixel values to [0, 255] for saving
                    img_array = np.array(img, dtype=np.uint8)

                    # Save as .png
                    save_path = os.path.join(class_dest_path, img_name.split('.')[0] + ".png")
                    Image.fromarray(img_array).save(save_path)

            logging.info(f"✅ Successfully transformed images from {source_dir} to {dest_dir}")

        except Exception as e:
            raise CustomException(e)
            


    def initiate_data_transformation(self):
        """
        Perform transformation on train, test, and validation datasets
        and save preprocessing object
        """
        try:
            # Transform train/test/validation sets
            self.transform_and_save_images(self.data_ingestion_artifacts.trained_file_path, self.data_transformation_config.transformation_train_dir)
            self.transform_and_save_images(self.data_ingestion_artifacts.tested_file_path, self.data_transformation_config.transformation_test_dir)
            self.transform_and_save_images(self.data_ingestion_artifacts.validated_file_path, self.data_transformation_config.transformation_validate_dir)

            

            logging.info(f"✅ Preprocessing object saved at: ")

            data_transformation_artifacts = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformation_train_dir,
                transformed_test_file_path= self.data_transformation_config.transformation_test_dir,
                transformed_validate_file_path=self.data_transformation_config.transformation_validate_dir,
                transformed_object_file_path= self.data_transformation_config.transformation_transformed_object_dir
            ) 


            return data_transformation_artifacts

        except Exception as e:
            raise CustomException(e)

          
        
        