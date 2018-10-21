
# Just a simple configs file with all the hyperparameters you need to set

# Note: 0 sets the seed to torch's initial seed
RANDOM_SEED = 10

# Training-specific hyperparameters
BATCH_SIZE = 20
CHECKPOINT = 20

# The name of the file in which the parameters are saved
FNAME = "model.pickled"

# First choose one of the configurations below
MODEL_CONFIG = 1

"""
---------------------
Define configurations.
---------------------
"""
### Configuration 1 ###
if MODEL_CONFIG == 1:
	EPOCHS = 25
	N_ITER_1 = 20
	N_ITER_2 = 4
	BETA = 0.5
	DELTA = 0.5
	LAYER_SIZES = [
		28 * 28,
		500,
		10,
	]
	LEARNING_RATES = [
		0.1,
		0.05,
	]

### Configuration 2 ###
elif MODEL_CONFIG == 2:
	EPOCHS = 60
	N_ITER_1 = 100
	N_ITER_2 = 6
	BETA = 1.0
	DELTA = 0.5
	LAYER_SIZES = [
		28 * 28,
		500,
		500,
		10,
	]
	LEARNING_RATES = [
		0.4,
		0.1,
		0.01,
	]


### Configuration 3 ###
elif MODEL_CONFIG == 3:
	EPOCHS = 160
	N_ITER_1 = 500
	N_ITER_2 = 8
	BETA = 1.0
	DELTA = 0.5
	LAYER_SIZES = [
		28 * 28,
		500,
		500,
		10,
	]
	LEARNING_RATES = [
		0.128,
		0.032,
		0.008,
		0.002,
	]

else:
	print("Error: invalid configuration:", MODEL_CONFIG)
	print("Please set MODEL_CONFIG to 1, 2, or 3 in training_config.py")
	exit()

