""" A module collecting parameters used in the algorithms. """

###################################
# Parameters for solution accuracy.
###################################

TOLERANCE_FOR_ARTIFICIAL_VARS = 1e-8
TOLERANCE_FOR_REDUCED_COSTS = 1e-6


#############################################################
# Parameters for the network crossover (CNET/TNET) algorithm.
#############################################################

# Set expanding ratio for column generation.
COLUMN_GENERATION_RATIO = 2


#######################################################
# Parameters for the perturbation crossover algorithm.
#######################################################
OPTIMAL_FACE_ESTIMATOR = 1e-3
OPTIMAL_FACE_ESTIMATOR_UPDATE_RATIO = 1e-5
PERTURB_THRESHOLD = 1e-6
CONSTANT_SCALE_FACTOR = 1e-2
PRIMAL_DUAL_GAP_THRESHOLD = 1e-8
PROJECTOR_THRESHOLD = 1e-8
PERTURB_UPPER_BOUND = 1e6
