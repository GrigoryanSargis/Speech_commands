from preprocessing import Preprocessing
import torchaudio
import torch
from config import config

# its a common practice to use some kind of feature extraction when working with audio
# make those features using dataset.map() method
# We suggest MFCC (Mel Frequency Cepstral Ceofficients) 

config_feature = config['feature']
#init is given and function names
class FeatureMappings:
    def __init__(self):
        self.sample_rate = config['sample_rate']
        self.output_feature = config['feature']
        # fft params
        window_size_ms = config_feature['window_size_ms']
        self.frame_length = int(self.sample_rate * window_size_ms)
        frame_step = config_feature['window_stride']
        self.frame_step = int(self.sample_rate * frame_step)
        assert (self.frame_step == self.sample_rate * frame_step), \
            'frame step,  must be integer '
        self.fft_length = config_feature['fft_length']
        # mfcc params
        self.lower_edge_hertz = config_feature['mfcc_lower_edge_hertz']
        self.upper_edge_hertz = config_feature['mfcc_upper_edge_hertz']
        self.num_mel_bins = config_feature['mfcc_num_mel_bins']
        self.num_mfccs = config_feature.get('mfcc_num_ceps', 13)

        """
        This is the main function
        it gets a preprocessing object instance as an argument and therefore has access to preprocessing.dataset
            add another mapping` we suggest MFCC, but you are free to use other feature, the other most common one is log mel spectrogram
        """
         
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.fft_length,
            win_length=self.frame_length,
            hop_length=self.frame_step,
            f_min=self.lower_edge_hertz,
            f_max=self.upper_edge_hertz,
            n_mels=self.num_mel_bins,
        )
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.num_mfccs,
            melkwargs={
                'n_fft': self.fft_length,
                'win_length': self.frame_length,
                'hop_length': self.frame_step,
                'f_min': self.lower_edge_hertz,
                'f_max': self.upper_edge_hertz,
                'n_mels': self.num_mel_bins,
            }
        )

    def create_features(self, preprocessing):
        """
        Creates feature datasets by applying feature transformations
        on the datasets from the preprocessing object.
        """
        train_dataset = self.map_features(preprocessing.train_loader)
        val_dataset = self.map_features(preprocessing.val_loader)
        test_dataset = self.map_features(preprocessing.test_loader)
        return train_dataset, val_dataset, test_dataset

    def map_features(self, data_loader):
        """
        Applies feature mapping to the given data loader.
        """
        transformed_data = []
        for audio, label in data_loader:
            mfccs = self.map_input_to_mfcc(audio)
            transformed_data.append((mfccs, label))
        return transformed_data


    def get_stft(self, audio):
        """
        Computes the Short-Time Fourier Transform (STFT) of the audio.
        """
        return torch.stft(
            audio,
            n_fft=self.fft_length,
            hop_length=self.frame_step,
            win_length=self.frame_length,
            return_complex=True,
        )

    def stft_to_log_mel_spectrogram(self, stft):
        """
        Converts the STFT to a log Mel spectrogram.
        """
        magnitude = torch.abs(stft)
        mel_spectrogram = self.mel_transform(magnitude)
        log_mel_spectrogram = torch.log1p(mel_spectrogram)
        return log_mel_spectrogram

    def map_input_to_mfcc(self, audio):
        """
        Computes MFCCs from the input audio.
        """
        mfccs = self.mfcc_transform(audio)
        if config['model_params'].get('data_normalization', False):
            mfccs = (mfccs - mfccs.mean(dim=-1, keepdim=True)) / mfccs.std(dim=-1, keepdim=True)
        return mfccs



