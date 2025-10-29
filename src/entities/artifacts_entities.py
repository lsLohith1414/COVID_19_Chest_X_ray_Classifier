from dataclasses import dataclass


@dataclass
class DataIngestionAftifacts:
    trained_file_path:str
    tested_file_path:str
    validated_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_validate_file_path: str



@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    test_ds:str
    class_names:str

@dataclass
class ModelEvaluationArtifact:
    acc:float
    precision:float
    recall:float
    f1:float


