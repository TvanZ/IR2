import os

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

# clearing output file if it already exists
source_file = "test.weights"
output_file = "test.padded_weights"
pad_weights(target_file=source_file, output_filename=output_file)


