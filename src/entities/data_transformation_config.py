from src.constants import data_transformation_constants
from src.entities.global_config import GlobalConfig
import os


class DataTransformationConfig:
    def __init__(self, global_config:GlobalConfig):

        self.transformation_dir = os.path.join(global_config.artifacts_dir, data_transformation_constants.DATA_TRANSFORMATION_DIR_NAME)

        self.transformed_data_dir = os.path.join(self.transformation_dir,data_transformation_constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, )

        self.transformation_train_dir = os.path.join(self.transformed_data_dir , global_config.train_data_dir)

        self.transformation_test_dir = os.path.join(self.transformed_data_dir, global_config.test_data_dir)

        self.transformation_validate_dir = os.path.join(self.transformed_data_dir, global_config.validate_data_dir)

        self.transformation_transformed_object_dir = os.path.join(self.transformation_dir, data_transformation_constants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR)

        


if __name__ == "__main__":

    global_config = GlobalConfig()

    obj = DataTransformationConfig(global_config=global_config)

    print(obj.transformation_dir)
    print(obj.transformed_data_dir)
    print(obj.transformation_train_dir)
    print(obj.transformation_test_dir)
    print(obj.transformation_validate_dir)
    print(obj.transformation_transformed_object_dir)