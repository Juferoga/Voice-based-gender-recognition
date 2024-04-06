import os
import math

class DataManager:
    def __init__(self, scream_path, non_scream_path):
        self.scream_path = scream_path
        self.non_scream_path = non_scream_path

    def make_folder(self, folder_path):
        try:
            os.mkdir(folder_path)
            print(folder_path, "ha sido creado ...")
        except FileExistsError:
            print(folder_path, "ya existe ...")
        except Exception as e:
            print("Exception raised: ", folder_path, "No se pudo crear porque ...", e)

    def move_files(self, src, dst, group):
        for fname in group:
            os.rename(os.path.join(src, fname), os.path.join(dst, fname))

    def split_data(self, files):
        training_data, testing_data = [], []
        length_data = len(files)
        length_separator = math.trunc(length_data * 2 / 3)
        training_data += files[:length_separator]
        testing_data += files[length_separator:]
        return training_data, testing_data

    def manage(self):
        # Gather file names
        scream_files = os.listdir(self.scream_path)
        non_scream_files = os.listdir(self.non_scream_path)

        # Split data into training and testing sets
        training_scream, testing_scream = self.split_data(scream_files)
        training_non_scream, testing_non_scream = self.split_data(non_scream_files)

        # Make training and testing folders
        self.make_folder("TrainingData")
        self.make_folder("TestingData")
        self.make_folder("TrainingData/screams")
        self.make_folder("TrainingData/non_screams")
        self.make_folder("TestingData/screams")
        self.make_folder("TestingData/non_screams")

        # Move files
        self.move_files(self.scream_path, "TrainingData/screams", training_scream)
        self.move_files(self.scream_path, "TestingData/screams", testing_scream)
        self.move_files(self.non_scream_path, "TrainingData/non_screams", training_non_scream)
        self.move_files(self.non_scream_path, "TestingData/non_screams", testing_non_scream)

if __name__ == "__main__":
    # Paths to the scream and non_scream directories
    scream_path = "scream"
    non_scream_path = "non_scream"

    data_manager = DataManager(scream_path, non_scream_path)
    data_manager.manage()
