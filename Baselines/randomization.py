import os
import numpy as np

# switching to filepaths so Bella can open test.weights
# just something Bella has to do, because of PyCharm settings
dir_filepath = os.path.join("Unbiased-Learning-to-Rank-with-Unbiased-Propensity-Estimation-master",
                                "Input_Data",
                                "test")
os.chdir(dir_filepath)

# TODO: pad AFTER randomization scores
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


def randomize_rankings(randomized_filename, RandomType="Top_RandN"):
    # clears file if it already exists
    open(randomized_filename, 'w').close()

    # loop through SVM_rankings line by linke
    with open(output_file) as SVM_rankings:
        query_rankings = SVM_rankings.readlines()

        for document_list in query_rankings:
            # making str of text into numpy array with rankings
            queryID = str(document_list).strip().split(' ')[0]
            doc_rankings = np.asarray(str(document_list).strip().split(' ')[1:])

            # shuffling top 5 documents randomly
            if RandomType == "Top_RandN":
                shuffles_inds = np.random.permutation(5)
                # adding the non-shuffled indicies back in
                shuffled_doc_rankings = np.append(doc_rankings[shuffles_inds], doc_rankings[5:])
                # adding the query ID back in
                shuffled_doc_rankings = np.append(queryID, shuffled_doc_rankings)
                # convert np array back into list
                shuffled_doc_rankings = shuffled_doc_rankings.tolist()
                shuffled_doc_rankings = " ".join(shuffled_doc_rankings) + '\n'

                # TODO: figure out a better place to put this?
                with open(randomized_filename, 'a') as randomized_file:
                    randomized_file.write(shuffled_doc_rankings)

            elif RandomType == "RandPairs":
                # TODO: pairwise
                print(RandomType)

            else:
                "ERROR! Unknown randomization type. Either input 'Top_RandN' or 'RandPairs' to randomize rankings."


if __name__ == "__main__":

    source_file = "test.weights"
    output_file = "test.padded_weights"
    pad_weights(target_file=source_file, output_filename=output_file)

    # types of randomization implemented
    options = ["Top_RandN", "RandPairs"]
    randomized_filename = "test.randomized_weights"
    randomize_rankings(randomized_filename, options[2])















