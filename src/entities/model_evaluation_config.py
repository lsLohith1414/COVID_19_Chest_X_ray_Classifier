from src.entities.global_config import GlobalConfig
from src.constants import model_evaluation_constants

import os


class ModelEvaluationConfig:
    def __init__(self,global_config:GlobalConfig):

        self.evaluation_dir:str = os.path.join(global_config.artifacts_dir, model_evaluation_constants.EVALUATION_DIR)

        self.matrics_file_name = os.path.join(self.evaluation_dir, model_evaluation_constants.METRICS_FILE_NAME)

        self.confusion_matrix_file_name:str = os.path.join(self.evaluation_dir, model_evaluation_constants.CONFUSION_MATRIX_FILE_NAME)