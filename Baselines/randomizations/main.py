import os
from Baselines.randomizations.random_utils import *
from Baselines.randomizations.randomize import randomize

# args for script
click_model_path = os.path.join('click_models', 'qd_train_position_biased_model.pickle')
randomTypes = ['randTopN', 'randPair']
selected_randomType = randomTypes[0]

# obtain shuffled
pickle_filename = randomize(click_model_path=click_model_path, selected_randomType=selected_randomType)

print_model(pickle_filename)
# read_results(pickle_filename)
#       (1) Get pickled filename to be returned
#       (2) Load file and access click decisions. Write code to access click decisions