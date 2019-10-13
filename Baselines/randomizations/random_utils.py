import pickle


def print_model(click_model_path):

    with open(click_model_path, 'rb') as click_model:
        click_model = pickle.load(click_model)

    for query in click_model.keys():
        docs_list = click_model[query]
        print(docs_list)
        break


def read_results(click_model_path):

    with open(click_model_path, 'rb') as click_model:
        click_model = pickle.load(click_model)

    for query in click_model.keys():
        docs_list = click_model[query]
        print(docs_list)
        break