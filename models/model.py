from abc import ABC, abstractmethod
import os
import json
import numpy as np


class Abstract_Model(ABC):
    """
    Class structure of a generic Topic Modelling implementation
    """

    hyperparameters = {}

    def __init__(self):
        """
        Create a blank model to initialize
        """

    def set_hyperparameters(self, **kwargs):
        """
        Set model hyperparameters
        """
        for key, value in kwargs.items():
            self.hyperparameters[key] = value

    @abstractmethod
    def train_model(self, dataset, hyperparameters, top_words=10, topic_word_matrix=True, topic_document_matrix=True):
        """
        Train the model.
        Return a dictionary with up to 3 entries,
        'topics', 'topic-word-matrix' and 'topic-document-matrix'.
        'topics' is the list of the most significative words for
        each topic (list of lists of strings).
        'topic-word-matrix' is an NxV matrix of weights where N is the number
        of topics and V is the vocabulary length.
        'topic-document-matrix' is an NxD matrix of weights where N is the number
        of topics and D is the number of documents in the corpus.

        """
        pass


def save_model_output(model_output, path=os.curdir, appr_order=7):
    """
    Saves the model output in the choosen directory

    Parameters
    ----------
    model_output: output of the model
    path: path in which the file will be saved and name of the file
    appr_order: approximation order (used to round model_output values)
    """

    to_save = {}
    for single_output in model_output.keys():
        if single_output != "topics" and single_output != "test-topics":
            to_save[single_output] = (
                model_output[single_output].round(appr_order))
        else:
            to_save[single_output] = (model_output[single_output])
    np.savez_compressed(
        path,
        **to_save)


def load_model_output(output_path, vocabulary_path=None, top_words=10):
    """
    Loads a model output from the choosen directory

    Parameters
    ----------
    output_path: path in which th model output is saved
    topics_path: path in which the vocabulary is saved
                 (optional, used to retrieve the top k words of each topic)
    top_words: top k words to retrieve for each topic
            (in case a vocabulary path is given)
    """
    output = dict(np.load(output_path))
    if vocabulary_path != None:
        file = open(vocabulary_path, "r")
        vocabulary = json.load(file)
        file.close()

        topics_output = []
        for topic in output["topic-word-matrix"]:
            top_k = np.argsort(topic)[-top_words:]
            top_k_words = list(reversed([vocabulary[str(i)] for i in top_k]))
            topics_output.append(top_k_words)

        output["topics"] = topics_output

    return output
