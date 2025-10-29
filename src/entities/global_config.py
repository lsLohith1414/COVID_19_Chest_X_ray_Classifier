from src.constants import global_constants
from datetime import datetime
import os

class GlobalConfig:
    def __init__(self, timestamp:datetime = None):

        if timestamp is None:
            timestamp = datetime.now()

        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M")
    
        artifact_name = global_constants.ARTIFACT_DIR_NAME
        self.artifacts_dir = os.path.join(artifact_name,timestamp)
        self.raw_data_dir = global_constants.RAW_DATA_DIR
        self.train_data_dir = global_constants.TRAIN_DATA_DIR
        self.test_data_dir = global_constants.TEST_DATA_DIR
        self.validate_data_dir = global_constants.VALIDATION_DATA_DIR
        self.timestamp = timestamp
        
