from src.entities.global_config import GlobalConfig 
from src.constants import data_ingestion_constants
import os

class DataIngestionConfig:
    def __init__(self, global_config:GlobalConfig):
        
        self.data_ingestion_dir:str = os.path.join(global_config.artifacts_dir, data_ingestion_constants.DATA_INGESTION_DIR_NAME)

        self.feature_store_dir: str = os.path.join(self.data_ingestion_dir, data_ingestion_constants.DATA_INGESTION_FEATURE_STORE_NAME, global_config.raw_data_dir)

        ingested_dir = os.path.join(self.data_ingestion_dir, data_ingestion_constants.DATA_INGESTION_INGESTED_NAME)

        self.ingested_train_dir:str = os.path.join(ingested_dir, global_config.train_data_dir)

        self.ingested_test_dir:str = os.path.join(ingested_dir, global_config.test_data_dir)

        self.ingested_validation_dir:str = os.path.join(ingested_dir, global_config.validate_data_dir)

        self.source_dir:str = os.path.join(os.getcwd(), data_ingestion_constants.DATA_SOURCE_DIR)

    

if __name__ == "__main__":

   

    obj = DataIngestionConfig(global_config=GlobalConfig())

    print(obj.data_ingestion_dir)
    print(obj.feature_store_dir)
    print(obj.ingested_train_dir)
    print(obj.ingested_test_dir)
    print(obj.ingested_validation_dir)
    print(obj.source_dir)

    print(os.listdir(obj.source_dir))

# python -m src.entities.data_ingestion_config