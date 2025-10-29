RAW_DATA_DIR: str = 'raw'
TRAIN_DATA_DIR: str = 'train'
TEST_DATA_DIR: str = 'test'
VALIDATION_DATA_DIR: str = "validation" 
ARTIFACT_DIR_NAME: str = 'Artifacts'

IMAGE_SIZE = (224, 224)                 # input image size for VGG19/ResNet50/DenseNet121
BATCH_SIZE = 32                          # default batch size
NUM_CLASSES = 3                          # COVID-19, Normal, Pneumonia
RANDOM_SEED = 42                         # reproducibility


# ---------------- Dataset Split Ratios ----------------
TRAIN_SPLIT_RATIO = 0.70
VALIDATION_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15