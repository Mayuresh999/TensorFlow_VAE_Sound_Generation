# import librosa
# import numpy as np
# import pickle
# import os

# class Loader:
#     ''' loads an audio file'''
#     def __init__(self, sample_rate, duration, mono):
#         self.sample_rate = sample_rate
#         self.duration = duration
#         self.mono = mono

#     def load(self, filepath):
#         signal = librosa.load(filepath, sr=self.sample_rate, duration=self.duration, mono=self.mono)

#         return signal

# class Padder:
#     '''adds padding to audio file'''
#     def __init__(self, mode = "constant"):
#         self.mode = mode

#     def left_pad(self, array, num_missing):
#         padded_array = np.pad(array, (num_missing, 0), self.mode)

#         return padded_array

#     def right_pad(self, array, num_missing):
#         print(array, num_missing)
#         padded_array = np.pad(array[0], (0, num_missing), self.mode)
        
#         return padded_array


# class LogSpectrogramExtractor:
#     '''extracts log spectrogram in decibles from audio file'''
#     def __init__(self, frame_size, hop_length):
#         self.frame_size = frame_size
#         self.hop_length = hop_length

#     def extract(self, signal):
#         stft = librosa.stft(signal, n_fft = self.frame_size, hop_length = self.hop_length)
#         spectrogram = np.abs(stft)
#         log_spectrogram = librosa.amplitude_to_db(spectrogram)
#         return log_spectrogram


# class MinMaxNormalizer:
#     '''Applies MinMax Normalizer to array'''
#     def __init__(self, min_val, max_val):
#         self.min_val = min_val
#         self.max_val = max_val

#     def normalize(self, array):
#         norm_array = (array - array.min()) / (array.max() - array.min())
#         norm_array = norm_array * (self.max_val - self.min_val) + self.min_val
#         return norm_array

#     def denormalize(self, norm_array, original_min, original_max):
#         array = (norm_array - self.min_val) / (self.max_val - self.min_val)
#         array = array * (original_max - original_min) + original_min    
#         return array


# class Saver:
#     '''saves features and min/max values '''
#     def __init__(self, feature_save_dir, min_max_val_save_dir):
#         self.feature_save_dir = feature_save_dir
#         self.min_max_val_save_dir = min_max_val_save_dir

#     def save_feature(self, feature, filepath):
#         save_path = self._generate_save_path(filepath)
#         np.save(save_path, feature)

#     def save_min_max_values(self, min_max_values):
#         save_path = os.path.join(self.min_max_val_save_dir, "min_max_values.pkl")
#         self._save(min_max_values, save_path)

#     def _generate_save_path(self, filepath):
#         file_name = os.path.split(filepath)[1]
#         save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
#         return save_path
    
#     @staticmethod
#     def _save(data, save_path):
#         with open(save_path, "wb") as f:
#             pickle.dump(data, f)



# class PreProcessingPipeline:
#     '''processes audio files in a dir applying following steps to each file'''

#     def __init__(self) -> None:
#         self.padder = None
#         self.extractor = None
#         self.normalizer = None
#         self.saver = None
#         self.min_max_values = {}
#         self._loader = None
#         self._num_expected_samples = None

#     @property
#     def loader(self) -> None:
#         return self._loader
    
#     @loader.setter
#     def loader(self, loader):
#         self._loader = loader
#         self._num_expected_samples = int(loader.sample_rate * loader.duration)

#     def process(self, audio_file_dir):
#         for root, _, files in os.walk(audio_file_dir):
#             for file in files:
#                 filepath = os.path.join(root, file)
#                 self._process_file(filepath)
#                 print("Processing file: %s" % filepath)
#         self.saver.save_min_max_values(self.min_max_values)
                
#     def _process_file(self, filepath):
#         signal = self.loader.load(filepath)
#         if self._is_padding_necessary(signal):
#             signal = self._apply_padding(signal)
#         feature = self.extractor.extract(signal)
#         norm_feature = self.normalizer.normalize(feature)
#         save_path = self.saver.save_feature(norm_feature, filepath)
#         self._store_min_max_value(save_path, feature.min(), feature.max())
    
#     def _is_padding_necessary(self, signal):
#         num_expected_samples = self._num_expected_samples

#         if len(signal) < num_expected_samples:
#             return True
#         return False
    
#     def _apply_padding(self, signal):
#         num_missing_samples =self._num_expected_samples - len(signal)
#         padded_signal = self.padder.right_pad(signal, num_missing_samples)

#         return padded_signal
    
#     def _store_min_max_value(self, save_path, min_val, max_val):
#         self.min_max_values[save_path] = {
#             "mix": min_val, "max":max_val
#         }



# if __name__ == "__main__":
#     FRAME_SIZE = 512
#     HOP_LENGTH = 256
#     DURATION = 0.74 # in seconds
#     SAMPLE_RATE = 22050
#     MONO = True

#     SPECTOGRAMS_SAVE_PATH = "SPECTOGRAMS"
#     MIN_MAX_VALUE_SAVE_DIR = "MIN_MAX_VALUES"
#     FILES_DIR = "data"

#     loader = Loader(sample_rate=SAMPLE_RATE, duration=DURATION, mono=MONO)
#     padder = Padder()
#     log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
#     min_max_normalizer = MinMaxNormalizer(0, 1)
#     saver = Saver(SPECTOGRAMS_SAVE_PATH, MIN_MAX_VALUE_SAVE_DIR)


#     processing_pipeline = PreProcessingPipeline()

#     processing_pipeline.loader = loader
#     processing_pipeline.padder = padder
#     processing_pipeline.extractor = log_spectrogram_extractor
#     processing_pipeline.normalizer = min_max_normalizer
#     processing_pipeline.saver = saver
    
    
#     processing_pipeline.process(FILES_DIR)





"""
1- load a file
2- pad the signal (if necessary)
3- extracting log spectrogram from signal
4- normalise spectrogram
5- save the normalised spectrogram

PreprocessingPipeline
"""
import os
import pickle

import librosa
import numpy as np


class Loader:
    """Loader is responsible for loading an audio file."""

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]
        return signal


class Padder:
    """Padder is responsible to apply padding to an array."""

    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    """LogSpectrogramExtractor extracts log spectrograms (in dB) from a
    time-series signal.
    """

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to an array."""

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


class Saver:
    """saver is responsible to save features, and the min max values."""

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir,
                                 "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path


class PreprocessingPipeline:
    """PreprocessingPipeline processes audio files in a directory, applying
    the following steps to each file:
        1- load a file
        2- pad the signal (if necessary)
        3- extracting log spectrogram from signal
        4- normalise spectrogram
        5- save the normalised spectrogram

    Storing the min max values for all the log spectrograms.
    """

    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }

if __name__ == "__main__":
    try:
        FRAME_SIZE = 512
        HOP_LENGTH = 256
        DURATION = 0.74  # in seconds
        SAMPLE_RATE = 22050
        MONO = True

        SPECTROGRAMS_SAVE_DIR = "SPECTOGRAMS"
        MIN_MAX_VALUES_SAVE_DIR = "MIN_MAX_VALUES"
        FILES_DIR = "data"

        # instantiate all objects
        loader = Loader(SAMPLE_RATE, DURATION, MONO)
        padder = Padder()
        log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
        min_max_normaliser = MinMaxNormaliser(0, 1)
        saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

        preprocessing_pipeline = PreprocessingPipeline()
        preprocessing_pipeline.loader = loader
        preprocessing_pipeline.padder = padder
        preprocessing_pipeline.extractor = log_spectrogram_extractor
        preprocessing_pipeline.normaliser = min_max_normaliser
        preprocessing_pipeline.saver = saver

        preprocessing_pipeline.process(FILES_DIR)
    except Exception as e:
        print(e)
        raise e