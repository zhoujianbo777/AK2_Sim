"""
feature_extractor.py  —  Per-channel EDI feature extraction for M1 model.

Single source of truth used by BOTH:
  - tools/gen_training_data.py  (offline training data generation)
  - modules/engine_ai.py        (online inference)

Feature vector (20-dim) per channel:
  [0]  dist_norm       edi_distance / 6.5
  [1]  amp             edi_amplitude  (raw, already 0-1 normalized at sensor)
  [2]  conf_norm       edi_confidence / 100.0
  [3]  echo_norm       edi_echo_type / 3.0
  [4]  env_max         envelope.max()
  [5]  env_mean        envelope.mean()
  [6]  env_std         envelope.std()
  [7]  peak_pos_norm   argmax(envelope) / 255.0
  [8]  env_rms         sqrt(mean(envelope²))
  [9]  half_width_norm len(indices where env >= peak/2) / 255.0
  [10] skewness        3rd standardised moment
  [11] kurtosis        4th standardised moment
  [12-19] bin0..bin7   mean of 8 equal-width envelope bins (32 pts each)

Reference: AK2 AI模型开发说明 §3.2.2 (M1 feature spec)
"""

import numpy as np


def extract_channel_features(
    ch: int,
    edi_dist:  "np.ndarray",   # shape (12,) float32
    edi_amp:   "np.ndarray",   # shape (12,) float32
    edi_conf:  "np.ndarray",   # shape (12,) uint8 or float32, range 0-100
    edi_echo:  "np.ndarray",   # shape (12,) uint8 or float32, range 0-3
    envelope:  "np.ndarray",   # shape (256,) float32
) -> np.ndarray:
    """Compute the 20-dim M1 input feature vector for a single sensor channel.

    This function is channel-independent (no cross-channel normalisation);
    the model shares weights across all 12 channels.
    """
    feat = np.zeros(20, dtype=np.float32)
    env  = envelope.astype(np.float32)

    feat[0] = float(edi_dist[ch]) / 6.5
    feat[1] = float(edi_amp[ch])
    feat[2] = float(edi_conf[ch]) / 100.0
    feat[3] = float(edi_echo[ch]) / 3.0

    feat[4] = float(env.max())
    feat[5] = float(env.mean())
    feat[6] = float(env.std())

    peak_pos = int(env.argmax())
    feat[7]  = peak_pos / 255.0
    feat[8]  = float(np.sqrt(np.mean(env ** 2)))   # RMS

    peak_val = env[peak_pos]
    above    = np.where(env >= peak_val * 0.5)[0]
    feat[9]  = float(len(above)) / 255.0

    env_std  = env.std() + 1e-8
    env_mean = env.mean()
    feat[10] = float(np.mean((env - env_mean) ** 3) / (env_std ** 3))   # skewness
    feat[11] = float(np.mean((env - env_mean) ** 4) / (env_std ** 4))   # kurtosis

    # 8 uniform bins, 32 points each (256 / 8 = 32)
    feat[12:20] = env.reshape(8, 32).mean(axis=1).astype(np.float32)

    return feat


def extract_all_channels(
    edi_dist:   "np.ndarray",   # (12,)
    edi_amp:    "np.ndarray",   # (12,)
    edi_conf:   "np.ndarray",   # (12,)
    edi_echo:   "np.ndarray",   # (12,)
    envelopes:  "np.ndarray",   # (12, 256)
) -> np.ndarray:
    """Compute the (12, 20) feature matrix for all channels in one frame."""
    feats = np.zeros((12, 20), dtype=np.float32)
    for ch in range(12):
        feats[ch] = extract_channel_features(
            ch, edi_dist, edi_amp, edi_conf, edi_echo, envelopes[ch]
        )
    return feats
