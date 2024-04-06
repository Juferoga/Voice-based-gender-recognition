import os
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings("ignore")


class GenderIdentifier:

    def __init__(self, screams_files_path, non_screams_files_path, screams_model_path, non_screams_model_path):
        self.screams_training_path = screams_files_path
        self.non_screams_training_path   = non_screams_files_path
        self.error                 = 0
        self.total_sample          = 0
        self.features_extractor    = FeaturesExtractor()
        # Verdaderos vs predicciones
        self.true_labels = []
        self.predictions = []
        # load models
        self.screams_gmm = pickle.load(open(screams_model_path, 'rb'))
        self.non_screams_gmm   = pickle.load(open(non_screams_model_path, 'rb'))

    def process(self):
        files = self.get_file_paths(self.screams_training_path, self.non_screams_training_path)
        print(files)
        # read the test directory and get the list of test audio files
        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            vector = self.features_extractor.extract_features(file)
            winner = self.identify_gender(vector)
            expected_gender = file.split("/")[1][:-1]

            print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
            self.true_labels.append(expected_gender == "scream")
            print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))
            self.predictions.append(winner == "scream")

            if winner != expected_gender: self.error += 1
            print("----------------------------------------------------")

        accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
        accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
        print(accuracy_msg)

        print("Matriz de confusión:")
        print(confusion_matrix(self.true_labels, self.predictions))
        print("\nReporte de clasificación:")
        print(classification_report(self.true_labels, self.predictions))

    def get_file_paths(self, screams_training_path, non_screams_training_path):
        # get file paths
        screams = [ os.path.join(screams_training_path, f) for f in os.listdir(screams_training_path) ]
        non_screams   = [ os.path.join(non_screams_training_path, f) for f in os.listdir(non_screams_training_path) ]
        files   = screams + non_screams
        return files

    def identify_gender(self, vector):
        # scream hypothesis scoring
        is_scream_scores         = np.array(self.screams_gmm.score(vector))
        is_scream_log_likelihood = is_scream_scores.sum()
        # non_scream hypothesis scoring
        is_non_scream_scores         = np.array(self.non_screams_gmm.score(vector))
        is_non_scream_log_likelihood = is_non_scream_scores.sum()

        print("%10s %5s %1s" % ("+ SCREAM SCORE",":", str(round(is_scream_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ NON SCREAM SCORE", ":", str(round(is_non_scream_log_likelihood,3))))

        if is_non_scream_log_likelihood > is_scream_log_likelihood: winner = "non_scream"
        else                                                : winner = "scream"
        return winner

if __name__== "__main__":
    gender_identifier = GenderIdentifier("TestingData/screams", "TestingData/non_screams", "screams.gmm", "non_screams.gmm")
    gender_identifier.process()
