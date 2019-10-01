import os
import numpy as np

# switching to filepaths so Bella can open test.weights
# just something Bella has to do, because of PyCharm settings
data_group = ["train", "test"]
# TODO: make some argument parser for this at some point?
selected_data_group = data_group[0]

dir_filepath = os.path.join("randomizations",
                                "Input_Data",
                                selected_data_group)
os.chdir(dir_filepath)


def randomize_rankings(randomized_filename, source_file, RandomType, topN=5):
    # clears file if it already exists
    open(randomized_filename, 'w').close()

    # loop through SVM_rankings line by linked
    with open(source_file) as SVM_rankings:
        query_rankings = SVM_rankings.readlines()

        for document_list in query_rankings:
            # making str of text into numpy array with rankings
            queryID = str(document_list).strip().split(' ')[0]
            doc_rankings = np.asarray(str(document_list).strip().split(' ')[1:])

            # throw an error if topN (to be shuffled) exceeds the number of
            # docs in the list
            assert topN < 10, "ERROR! Trying to shuffle more documents than are present in the ranking."

            # shuffling methods
            if RandomType == "RandTopN": # shuffling top 5 documents randomly
                if len(doc_rankings) >= 5:
                    shuffled_inds = np.random.permutation(topN)
                else:
                    shuffled_inds = np.random.permutation(len(doc_rankings))

            elif RandomType == "RandPair": # shuffling pairwise
                if len(doc_rankings) >= 5:
                    shuffled_inds = np.arange(topN)
                else:
                    shuffled_inds = np.arange(len(doc_rankings))

                # shuffling inds pair-wise
                for ind in range(len(shuffled_inds)-1):
                    shuffle_docs = np.random.choice([True, False])
                    if shuffle_docs:
                        shuffled_inds[ind], shuffled_inds[ind+1] = shuffled_inds[ind+1], shuffled_inds[ind]
            else:
                "ERROR! Unknown randomization type. Either input 'Top_RandN' or 'RandPairs' to randomize rankings."

            # adding the non-shuffled indicies back in
            shuffled_doc_rankings = np.append(doc_rankings[shuffled_inds], doc_rankings[5:])

            # adding the query ID back in
            shuffled_doc_rankings = np.append(queryID, shuffled_doc_rankings)
            # convert np array back into list
            shuffled_doc_rankings = shuffled_doc_rankings.tolist()
            shuffled_doc_rankings = " ".join(shuffled_doc_rankings) + '\n'

            with open(randomized_filename, 'a') as randomized_file:
                randomized_file.write(shuffled_doc_rankings)

if __name__ == "__main__":

    # todo: run script on train.weights
    #        rename swapped weights so .weights (one at a time b/c otherwise bugs)
    #                      --->  can only have one .weights file in a folder at a time
    #        in main.py
    # todo: put out, per position of click:
    #       run generate_click_data.py
    #       how likely it is to be observed per position

    source_file = selected_data_group + ".weights"
    # pad_weights(target_file=source_file, output_filename=output_file)

    # types of randomization implemented
    options = ["RandTopN", "RandPair"]
    selected_option = options[0]

    # creating filename
    randomized_filename = "{}.randomized_{}_weights".format(selected_data_group, selected_option)
    randomize_rankings(randomized_filename, source_file, selected_option)















