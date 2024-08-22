import librosa
import numpy as np
from scipy.stats import skew, kurtosis


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)
    mfcc_skew = skew(mfcc, axis=1)
    mfcc_kurtosis = kurtosis(mfcc, axis=1)

    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    spectral_flatness_mean = np.mean(spectral_flatness)
    spectral_rolloff_mean = np.mean(spectral_rolloff)
    chroma_stft_mean = np.mean(chroma_stft, axis=1)
    tonnetz_mean = np.mean(tonnetz, axis=1)

    zcr_mean = np.mean(zcr)
    rmse_mean = np.mean(rmse)

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_var,
            mfcc_skew,
            mfcc_kurtosis,
            [
                spectral_centroid_mean,
                spectral_bandwidth_mean,
                spectral_flatness_mean,
                spectral_rolloff_mean,
            ],
            spectral_contrast_mean,
            chroma_stft_mean,
            tonnetz_mean,
            [zcr_mean, rmse_mean],
        ]
    )

    return features
