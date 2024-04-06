import os
import pickle
import warnings
import numpy as np
from sklearn.mixture import GaussianMixture
from FeaturesExtractor import FeaturesExtractor

warnings.filterwarnings("ignore")


class ModelsTrainer:

    def __init__(self, screams_files_path, non_screams_files_path):
        self.screams_training_path = screams_files_path
        self.non_screams_training_path   = non_screams_files_path
        self.features_extractor    = FeaturesExtractor()

    def process(self):
        screams, non_screams = self.get_file_paths(self.screams_training_path,
                                            self.non_screams_training_path)
        # collect voice features
        scream_voice_features = self.collect_features(screams)
        non_scream_voice_features   = self.collect_features(non_screams)
        # generate gaussian mixture models
        screams_gmm = GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag', n_init = 3)
        non_screams_gmm   = GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag', n_init = 3)
        # fit features to models
        screams_gmm.fit(scream_voice_features)
        non_screams_gmm.fit(non_scream_voice_features)
        # save models
        self.save_gmm(screams_gmm, "screams")
        self.save_gmm(non_screams_gmm,   "non_screams")

    def get_file_paths(self, screams_training_path, non_screams_training_path):
        # get file paths
        screams = [ os.path.join(screams_training_path, f) for f in os.listdir(screams_training_path) ]
        non_screams   = [ os.path.join(non_screams_training_path, f) for f in os.listdir(non_screams_training_path) ]
        return screams, non_screams

    def collect_features(self, files):
        """
            Collect voice features from various speakers of the same gender.
            
            Args:
                files (list) : List of voice file paths.
                
            Returns:
                (array) : Extracted features matrix.
        """
        features = np.asarray(())
        # extract features for each speaker
        for file in files:
            print("%5s %10s" % ("PROCESSNG ", file))
            # extract MFCC & delta MFCC features from audio
            vector    = self.features_extractor.extract_features(file)
            # stack the features
            if features.size == 0:  features = vector
            else:                   features = np.vstack((features, vector))
        return features

    def save_gmm(self, gmm, name):
        """ Save Gaussian mixture model using pickle.

            Args:
                gmm        : Gaussian mixture model.
                name (str) : File name.
        """
        filename = name + ".gmm"
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print ("%5s %10s" % ("SAVING", filename,))


if __name__== "__main__":
    models_trainer = ModelsTrainer("TrainingData/screams", "TrainingData/non_screams")
    models_trainer.process()
