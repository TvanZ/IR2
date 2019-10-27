import os
from Baselines.randomizations.random_utils import *
from Baselines.randomizations.randomize import randomize
from Baselines.randomizations.calc_results import calc_values

# args for script

click_model_kind = ['simple_model', 'position_biased_model']  # , 'cascade_model']
click_model_path = os.path.join('click_models', 'qd_test_simple_model.pickle')
randomTypes = ['randTopN', 'randPair']
trial_num = 100

for counter in range(trial_num):
    # print('counter', counter)
    pickle_filename = os.path.join('outputs', randomize(click_model_path=click_model_path,
                                                        selected_randomType=randomTypes[1],
                                                        click_simulation_method=click_model_kind[1]))
    if counter == 0:
        randomized_results = read_results(pickle_filename, trial_num=counter)
    else:
        randomized_results = read_results(pickle_filename,
                                          trial_num=counter,

                                         randomized_results=randomized_results)

save_results(randomized_results, 'randomized_results_'+click_model_kind[1]+'.pickle')
# very messy, but Traian's code for calculating results
calc_values('randomized_results_'+click_model_kind[1]+'.pickle')


