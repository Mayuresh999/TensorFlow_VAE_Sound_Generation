from preprocessing import MinMaxNormaliser
import librosa

class SoundGenerator:
    '''responsible for generating audio from spectrogram'''
    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normalizer = MinMaxNormaliser(0, 1)

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representation = self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)

        return signals, latent_representation
    

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            # reshape log spectrogram
            log_spectrogram = spectrogram[:, :, 0]
            # apply denormalization
            denorm_log_spec = self._min_max_normalizer.denormalize(log_spectrogram, min_max_value["min"], min_max_value["max"])
            # log spectogram -> spectrogram
            spec = librosa.db_to_amplitude(denorm_log_spec)
            # apply griffin-lim and appent to signals
            signal = librosa.istft(spec, hop_length=self.hop_length)
            signals.append(signal)

        return signals
    

