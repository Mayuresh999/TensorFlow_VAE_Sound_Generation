import os
import numpy as np
import soundfile as sf
from sound_generator import SoundGenerator
from variational_autoencoder import VAE
import pickle
from train_vae import SPECTOGRAMS 

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
MIN_MAX_VALUES_PATH = "MIN_MAX_VALUES/min_max_values.pkl"

def load_dataset(data_path):
    x_train = []
    file_paths = []
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            spectrogram = np.load(file_path)
            x_train.append(spectrogram)
            file_paths.append(file_path)
            # print(spectrogram, spectrogram.shape)
    x_train = np.array(x_train[0])
    x_train = x_train[..., np.newaxis]
    print(x_train.shape)
    return x_train, file_paths

def select_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms = 2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrograms = spectrograms[sampled_indexes]
    file_paths = [file_paths[i] for i in sampled_indexes]
    print(file_paths)
    sampled_min_max_values = [min_max_values[j] for j in file_paths]

    print(sampled_min_max_values)
    return sampled_spectrograms, sampled_min_max_values



def save_signals(signals, save_dir, sample_rate = 22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


if __name__ == "__main__":
    # Initialize Sound Generator
    vae = VAE.load("model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # Load spectrograms + min_max_values
    with open(MIN_MAX_VALUES_PATH, 'rb') as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_dataset(SPECTOGRAMS)
    # Sample spectrograms + min_max_values
    sampled_specs, sampled_min_max = select_spectrograms(specs, file_paths, min_max_values, 5)
    # generate audio for sampled spectrograms
    signals, _ = sound_generator.generate(sampled_specs, sampled_min_max)
    # convert spectrogram samples to audio
    original_signals = sound_generator.convert_spectrograms_to_audio(sampled_specs, sampled_min_max)
    # save audio signals
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)
