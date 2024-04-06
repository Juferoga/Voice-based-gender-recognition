import os

if __name__== "__main__":

    # extract and manage dataset files
    print("# Mange and organize files")
    os.system('python3 Code/DataManager.py')

    # train gender gmm models
    print("# Train gender models")
    os.system('python3 Code/ModelsTrainer.py')

    # test system and recognise/identify speakers gender
    print(" # Identify genders")
    os.system('python3 Code/ScreamIdentifier.py')
