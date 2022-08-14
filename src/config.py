NUM_DEVICE = "0"

#prepare data
PATH_TO_TRAIN_IMAGES = "../train_data/slices_train"
PATH_TO_ANNOTATION = "../train_data/annotation_train.pkl"


#HM train data
PATH_TO_DATA = '../data.csv'
PATH_TO_DUMP_HM = "../train_data/heatmaps"



#dataloader
N = 6
M = 9
RESIZE_SHAPE = (512,512)
SEED = 777
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_VAL = 16

BATCH_SIZE_TEST = 1

#model
ENCODER = 'timm-efficientnet-b4'
DECODER = 'MAnet'
lr = 5.748492383047837e-05
BEST_MODEL = "best_model/best.pth"

#optuna
NUM_EPOCHS = 500
PATH_TO_EXPERIMENTS = "/storage1/ryazantsev/lumbar_spine/experiments/"
PATINENCE = 25

N_STARTUP_TRIALS = 15
N_WARMUP_STEPS = 25

N_TRIALS = 200


