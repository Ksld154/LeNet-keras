
## Global Constants
BATCH_SIZE = 64
PRE_EPOCHS = 2
TENSOR_TRANSMISSION_TIME = 30
FREEZE_OPTIONS = [0, 2, 4, 6, 7]
MOVING_AVERAGE_WINDOW_SIZE = 7


# main.py
INITIAL_FREEZE_LAYERS = 3
LOSS_DIFF_THRESHOLD = 0.05

# trainer.py
LOSS_COVERGED_THRESHOLD = 0.05
LOSS_THRESHOLD = 1.0
LOSS_THRESHOLD_ALPHA = 1.35
MAGIC_ALPHA = 0.3
