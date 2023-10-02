import pyaudio
import numpy as np
import os
import pickle
import warnings
from python_speech_features import mfcc
from python_speech_features import delta
from sklearn import preprocessing


# Configura la captura de audio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)


warnings.filterwarnings("ignore")

class FeatureExtractor:
    def __init__(self):
        # Configura la captura de audio
        self.audio_stream = pyaudio.PyAudio()
        self.audio_input_stream = self.audio_stream.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    def extract_features_realtime(self):
        """
        Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        from real-time audio using the python_speech_features module, performs Cepstral Mean
        Normalization (CMS) and combine it with MFCC deltas and the MFCC double
        deltas.
        Returns: 	    
            (array) : Extracted features matrix. 	
        """
        try:
            # Inicializa una lista para almacenar las características en tiempo real
            features_realtime = []

            while True:
                # Captura audio en tiempo real
                audio_data = self.audio_input_stream.read(1024)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                # Calcula las características MFCC en tiempo real
                mfcc_feature = mfcc(
                    audio_array,
                    44100,  # Tasa de muestreo
                    winlen=0.05,
                    winstep=0.01,
                    numcep=13,
                    nfilt=26,
                    nfft=512,
                    appendEnergy=True
                )

                # Realiza la normalización CMS
                mfcc_feature = preprocessing.scale(mfcc_feature)
                
                # Calcula las derivadas (deltas) en tiempo real
                deltas = delta(mfcc_feature, 2)
                double_deltas = delta(deltas, 2)

                # Combina todas las características en una sola matriz
                combined = np.hstack((mfcc_feature, deltas, double_deltas))

                # Agrega las características en tiempo real a la lista
                features_realtime.append(combined)

                # Puedes realizar el reconocimiento de género en tiempo real aquí si lo deseas

        except KeyboardInterrupt:
            # Maneja la interrupción del usuario (Ctrl+C)
            print("Proceso interrumpido por el usuario")

        finally:
            # Cierra el flujo de audio al finalizar
            self.audio_input_stream.stop_stream()
            self.audio_input_stream.close()
            self.audio_stream.terminate()

        return features_realtime

class GenderIdentifier:
    def __init__(self, females_path, males_path):
        self.error = 0
        self.total_sample = 0
        self.features_extractor = FeatureExtractor()
        # load models
        
        self.females_gmm = pickle.load(open(females_path, 'rb'))
        self.males_gmm = pickle.load(open(males_path, 'rb'))

    def process_real_time(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,  # Sample rate
            input=True,
            frames_per_buffer=1024
        )

        try:
            while True:
                audio_data = stream.read(1024)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                self.total_sample += 1

                # vector = self.features_extractor.extract_features_realtime(audio_array)
                vector = self.features_extractor.extract_features_realtime()
                winner = self.identify_gender(vector)

                os.system('clear')  # Limpiar la terminal antes de mostrar nuevos resultados

                print("Género actual:", winner)
                print("Número de muestras procesadas:", self.total_sample)
                print("Errores de género:", self.error)

        except KeyboardInterrupt:
            # Maneja la interrupción del usuario (Ctrl+C)
            print("Proceso interrumpido por el usuario")

        finally:
            # Cierra el flujo de audio y termina la captura
            stream.stop_stream()
            stream.close()
            p.terminate()

    def identify_gender(self, vector):
        # female hypothesis scoring
        is_female_scores = np.array(self.females_gmm.score(vector))
        is_female_log_likelihood = is_female_scores.sum()
        # male hypothesis scoring
        is_male_scores = np.array(self.males_gmm.score(vector))
        is_male_log_likelihood = is_male_scores.sum()

        if is_male_log_likelihood > is_female_log_likelihood:
            winner = "male"
        else:
            winner = "female"

        return winner

if __name__ == "__main__":
    gender_identifier = GenderIdentifier("/home/juferoga/repos/personal/Voice-based-gender-recognition/females.gmm", "/home/juferoga/repos/personal/Voice-based-gender-recognition/males.gmm")
    gender_identifier.process_real_time()

feat_ext=FeatureExtractor()
gen_idnt=GenderIdentifier()

try:
    while True:
        # Captura audio en tiempo real
        audio_data = stream.read(1024)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Extrae características de audio en tiempo real
        features = feat_ext.extract_features_realtime(audio_array)

        # Realiza el reconocimiento de género en tiempo real
        gender = gen_idnt.identify_gender(features)

        # Muestra el resultado por la terminal
        print(f"Género: {gender}")

except KeyboardInterrupt:
    # Maneja la interrupción del usuario (Ctrl+C)
    print("Proceso interrumpido por el usuario")

finally:
    # Cierra el flujo de audio y termina la captura
    stream.stop_stream()
    stream.close()
    p.terminate()

