import os
from Baselines.randomizations.random_utils import *
from Baselines.randomizations.randomize import randomize

# args for script
click_model_kind = ['POSITION_BIASED_MODEL', 'CASCADE_MODEL']
click_model_path = os.path.join('click_models', 'qd_train_position_biased_model.pickle')
randomTypes = ['randTopN', 'randPair']
trial_num = 3

for counter in range(trial_num):
    print('counter', counter)
    pickle_filename = os.path.join('outputs', randomize(click_model_path=click_model_path,
                                                        selected_randomType=randomTypes[0],
                                                        click_simulation_method=click_model_path[0]))
    if counter == 0:
        randomized_results = read_results(pickle_filename, trial_num=counter)
    else:
        randomized_results = read_results(pickle_filename,
                                          trial_num=counter,
                                          randomized_results=randomized_results)

save_results(randomized_results, 'randomized_results')



