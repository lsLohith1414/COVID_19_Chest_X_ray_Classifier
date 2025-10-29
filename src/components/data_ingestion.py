from src.entities.data_ingestion_config import DataIngestionConfig
from src.entities.global_config import GlobalConfig
from src.constants.global_constants import TRAIN_SPLIT_RATIO, VALIDATION_SPLIT_RATIO
from src.entities.artifacts_entities import DataIngestionAftifacts

from src.exception.exception import CustomException
from src.logging.logger import logging

import os
import shutil
import random


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise CustomException(e)
        
    def copy_raw_data(self):

        try:
            source_dir = self.data_ingestion_config.source_dir
            dest_dir = self.data_ingestion_config.feature_store_dir
            os.makedirs(dest_dir,exist_ok=True)


            for class_name in os.listdir(source_dir):
                class_source_path = os.path.join(source_dir, class_name)
                class_dest_path = os.path.join(dest_dir, class_name)
                os.makedirs(class_dest_path, exist_ok=True)

                for img_file in os.listdir(class_source_path):
                    shutil.copy(
                        os.path.join(class_source_path, img_file),
                        os.path.join(class_dest_path, img_file)
                    )

            logging.info("Copied raw images to features_store/raw")


        except Exception as e:
            raise CustomException(e)
        
    
    def split_data_as_train_test_validate(self):
        """Split images into train, validation, test and save in ingested folder"""
        try:
            random.seed(42)  # reproducibility

            # check if source exists
            if not os.path.exists(self.data_ingestion_config.feature_store_dir):
                raise FileNotFoundError(f"Feature store not found: {self.data_ingestion_config.feature_store_dir}")

            # iterate over each class
            for class_name in os.listdir(self.data_ingestion_config.feature_store_dir):
                class_path = os.path.join(self.data_ingestion_config.feature_store_dir, class_name)

                if not os.path.isdir(class_path):
                    continue  # skip non-folder files

                images = [
                    f for f in os.listdir(class_path)
                    if os.path.isfile(os.path.join(class_path, f))
                ]

                random.shuffle(images)
                total = len(images)
                train_end = int(total * TRAIN_SPLIT_RATIO)
                val_end = train_end + int(total * VALIDATION_SPLIT_RATIO)

                splits = {
                    self.data_ingestion_config.ingested_train_dir: images[:train_end],
                    self.data_ingestion_config.ingested_validation_dir: images[train_end:val_end],
                    self.data_ingestion_config.ingested_test_dir: images[val_end:],
                }

                for dest_dir, img_list in splits.items():
                    class_dest_dir = os.path.join(dest_dir, class_name)
                    os.makedirs(class_dest_dir, exist_ok=True)

                    for img_name in img_list:
                        src_img_path = os.path.join(class_path, img_name)
                        dest_img_path = os.path.join(class_dest_dir, img_name)
                        if not os.path.exists(dest_img_path):
                            os.system(f'copy "{src_img_path}" "{dest_img_path}"')

            logging.info("✅ Successfully split images into train, validation, and test sets (raw images).")


        except Exception as e:
            raise CustomException(e)
        


    def initiate_data_ingestion(self):

        try:
            self.copy_raw_data()
            self.split_data_as_train_test_validate()

            data_ingestion_artifacts = DataIngestionAftifacts(
                trained_file_path= self.data_ingestion_config.ingested_train_dir,
                tested_file_path=  self.data_ingestion_config.ingested_test_dir,
                validated_file_path= self.data_ingestion_config.ingested_validation_dir
            )

            return data_ingestion_artifacts
            


        except Exception as e:
            raise CustomException(e)
    

# if __name__ == "__main__":

   

#     obj = DataIngestionConfig(global_config=GlobalConfig())

#     ing = DataIngestion(data_ingestion_config=obj)

#     ing.copy_raw_data()

#     ing.split_data_as_train_test_validate()

# # python -m src.entities.data_ingestion_config

        