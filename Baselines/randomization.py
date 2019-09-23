import os
import numpy as np

# switching to filepaths so Bella can open test.weights
# just something Bella has to do, because of PyCharm settings
dir_filepath = os.path.join("Unbiased-Learning-to-Rank-with-Unbiased-Propensity-Estimation-master",
                                "Input_Data",
                                "test")
os.chdir(dir_filepath)

def pad_weights(target_file, output_filename):
    """
    Creates padded version test.padded_weights file in same dir.
    :param output_filename: desired output filename,
    :return: None
    """
    # clear file if it already exists
    open(output_filename, 'w').close()

    # read test.weights and make new version with padding
    # first creating padding of -1.0
    with open(target_file) as read_file:  # Use file to refer to the file object
        query_rankings = read_file.readlines()
        for document_list in query_rankings:
            # creating output file with padded weights
            with open(output_file, 'a') as write_file:
                docs_num = len(str(document_list).strip().split(' '))
                if docs_num < 11:
                    padding_len = 11 - docs_num
                    document_list = document_list.strip() + ' -1.0' * padding_len + '\n'
                    # Error check to know that things are being padded correctly
                    assert len(str(document_list).strip().split(' ')) == 11, 'ERROR: Padding done incorrectly!'
                write_file.write(document_list)


def randomize_rankings(randomized_filename, RandomType="Top_RandN", topN=5):
    # clears file if it already exists
    open(randomized_filename, 'w').close()

    # loop through SVM_rankings line by linke
    with open(output_file) as SVM_rankings:
        query_rankings = SVM_rankings.readlines()

        for document_list in query_rankings:
            # making str of text into numpy array with rankings
            queryID = str(document_list).strip().split(' ')[0]
            doc_rankings = np.asarray(str(document_list).strip().split(' ')[1:])

            # throw an error if topN (to be shuffled) exceeds the number of
            # docs in the list
            assert topN < 10, "ERROR! Trying to shuffle more documents than a present in the ranking."

            # shuffling methods
            if RandomType == "RandTopN": # shuffling top 5 documents randomly
                shuffled_inds = np.random.permutation(topN)

            elif RandomType == "RandPair": # shuffling pairwise
                shuffled_inds = np.arange(topN)
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

    source_file = "test.weights"
    output_file = "test.padded_weights"
    # TODO: fix order of padding with randomization
    pad_weights(target_file=source_file, output_filename=output_file)

    # types of randomization implemented
    options = ["RandTopN", "RandPair"]
    selected_option = options[1]

    # creating filename
    randomized_filename = "test.randomized_{}_weights".format(selected_option)
    randomize_rankings(randomized_filename, selected_option)















