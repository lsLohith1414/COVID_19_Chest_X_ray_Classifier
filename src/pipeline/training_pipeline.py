

from src.entities.data_ingestion_config import DataIngestionConfig
from src.entities.global_config import GlobalConfig
from src.components.data_ingestion import DataIngestion

from src.entities.artifacts_entities import DataIngestionAftifacts

from src.exception.exception import CustomException
from src.logging.logger import logging


from src.components.data_transformation import DataTransformation
from src.entities.data_transformation_config import DataTransformationConfig


from src.components.model_training import ModelTrainer
from src.entities.model_training_config import ModelTrainingConfig
from src.entities.artifacts_entities import ModelTrainerArtifact, DataTransformationArtifact, ModelEvaluationArtifact,DataIngestionAftifacts

from src.entities.model_evaluation_config import ModelEvaluationConfig
from src.components.model_evaluation import ModelEvaluation




class TrainingPipline:

    def __init__(self, global_config:GlobalConfig):
        
        try:
            self.global_config = global_config
        except Exception as e:
            raise CustomException(e)
        

    def start_data_ingestion(self):
        try:

            data_ingestion_config = DataIngestionConfig(global_config=self.global_config)

            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

            logging.info("Data ingestion started")

            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed successfully")

            logging.info(f"Paths: {data_ingestion_artifacts}")

            return data_ingestion_artifacts
        
        except Exception as e:
            raise CustomException(e) 
        



    def start_data_transformation(self, data_ingestion_artifacts : DataIngestionAftifacts):
        try:

            logging.info(f"Data transformation started")

            data_transformation_config = DataTransformationConfig(global_config=self.global_config)

            data_transformation = DataTransformation(data_ingestion_artifacts=data_ingestion_artifacts, data_transformation_config=data_transformation_config)

            data_transformation_artifacts = data_transformation.initiate_data_transformation()

            logging.info(f"Data transformation completed")

            logging.info(f"Data transformation Paths: {data_transformation_artifacts}")

            return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e)
        


    def start_model_trainer(self, data_transformation_artifacts:DataTransformationArtifact)->ModelTrainerArtifact:

        try:
            model_training_config = ModelTrainingConfig(global_config=self.global_config)

            model_train = ModelTrainer(model_training_config=model_training_config, model_transformation_artifact=data_transformation_artifacts)

            model_training_artifacts = model_train.initiate_model_training()

            logging.info(f"Model training Arifacts : {model_training_artifacts}")

            return model_training_artifacts
        except Exception as e:
            raise CustomException(e)
            

    def start_model_evaluation(self, model_training_artifacts:ModelTrainerArtifact):

        try:
            model_evaluation_config = ModelEvaluationConfig(global_config=self.global_config)

            model_eval = ModelEvaluation(model_evaluation_config=model_evaluation_config,
                                            model_trainer_artifact=model_training_artifacts)
                
            model_evaluation_artifacts = model_eval.initiate_evaluation()

            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e)
            
        


    def run_training_pipline(self):

        try:
            data_ingestion_artifacts = self.start_data_ingestion()

            data_transformation_artifacts = self.start_data_transformation(data_ingestion_artifacts=data_ingestion_artifacts)

            model_trainer_artifacts = self.start_model_trainer(data_transformation_artifacts=data_transformation_artifacts)

            model_evauation_artifacts = self.start_model_evaluation(model_training_artifacts=model_trainer_artifacts)

            return model_evauation_artifacts


        except Exception as e:
            raise CustomException(e)