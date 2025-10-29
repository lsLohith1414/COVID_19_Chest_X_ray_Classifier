from src.constants import model_training_constants
from src.entities.global_config import GlobalConfig
import os




class ModelTrainingConfig:
    def __init__(self, global_config:GlobalConfig):
        
        model_trainer_dir = os.path.join(global_config.artifacts_dir, model_training_constants.MODEL_TRAINER_DIR_NAME)

        self.trained_model_file_path = os.path.join(model_trainer_dir, model_training_constants.TRAINED_MODEL_DIR, model_training_constants.MODEL_FILE_NAME)


        