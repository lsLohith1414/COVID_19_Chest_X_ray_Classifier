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
from src.entities.artifacts_entities import ModelTrainerArtifact, DataTransformationArtifact

from src.entities.model_evaluation_config import ModelEvaluationConfig
from src.components.model_evaluation import ModelEvaluation



if __name__ == "__main__":
    try:
        if __name__ == "__main__":

            global_config=GlobalConfig()

            data_ingestion_config = DataIngestionConfig(global_config=global_config)

            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

            logging.info("Data ingestion started")

            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed successfully")

            print(data_ingestion_artifacts)

            logging.info(f"Paths: {data_ingestion_artifacts}")
            




            # Data Transformation

            logging.info(f"Data transformation started")

            data_transformation_config = DataTransformationConfig(global_config=global_config)

            data_transformation = DataTransformation(data_ingestion_artifacts=data_ingestion_artifacts, data_transformation_config=data_transformation_config)

            data_transformation_artifacts = data_transformation.initiate_data_transformation()

            logging.info(f"Data transformation completed")

            logging.info(f"Data transformation Paths: {data_transformation_artifacts}")



            # Model training 

            model_training_config = ModelTrainingConfig(global_config=global_config)

            model_train = ModelTrainer(model_training_config=model_training_config, model_transformation_artifact=data_transformation_artifacts)

            model_training_artifacts = model_train.initiate_model_training()

            logging.info(f"Model training Arifacts : {model_training_artifacts}")



            # Model evaluation 

            model_evaluation_config = ModelEvaluationConfig(global_config=global_config)

            model_eval = ModelEvaluation(model_evaluation_config=model_evaluation_config, model_trainer_artifact=model_training_artifacts)
            
            model_evaluation_artifacts = model_eval.initiate_evaluation()

            
        

    except Exception as e:
        raise CustomException(e)