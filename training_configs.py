
# Just a simple configs file with all the hyperparameters you need to set

# Note: 0 sets the seed to torch's initial seed
RANDOM_SEED = 10

# Training-specific hyperparameters
BATCH_SIZE = 17

# EqProp specific hyperparameters
BETA = 1
DELTA = 0.5  # called epsilon in the paper
RHO = lambda x: x.clamp(0, 1)

# Choose one of the configurations below
CONFIGURATION = 2


### Configuration 1 ###
if CONFIGURATION == 1:
	N_ITER_1 = 20
	N_ITER_2 = 4
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
if CONFIGURATION == 2:
	N_ITER_1 = 100
	N_ITER_2 = 6
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
if CONFIGURATION == 3:
	N_ITER_1 = 500
	N_ITER_2 = 8
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




