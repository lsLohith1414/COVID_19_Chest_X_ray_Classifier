# ==============================================
# Model Training Constants
# ==============================================

MODEL_TRAINER_DIR_NAME = "model_trainer"
TRAINED_MODEL_DIR = "trained_model"
MODEL_NAME = "vgg19"
MODEL_FILE_NAME = "vgg19_final_model.keras"


# Model Parameters
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 3
DROPOUT_RATE = 0.5


# Training Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 1
BATCH_SIZE = 16
LOSS_FUNCTION = "sparse_categorical_crossentropy"
OPTIMIZER = "adam"
METRICS = ["accuracy"]
AUGMENT = True

# Regularization and Early Stopping
EARLY_STOPPING_PATIENCE = 5

# Reproducibility
RANDOM_SEED = 42


