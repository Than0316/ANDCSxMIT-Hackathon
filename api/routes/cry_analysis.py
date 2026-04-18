import os
import io
import json
import tempfile
import numpy as np
import librosa
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from scipy import stats
from scipy.signal import find_peaks
import soundfile as sf
import matplotlib.pyplot as plt
import uuid
from datetime import datetime
from pydub import AudioSegment

router = APIRouter()

# Create archives directory
ARCHIVES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'archives')
os.makedirs(ARCHIVES_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# ACOUSTIC FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features(y, sr):
    """Extract comprehensive acoustic features from audio signal."""
    features = {}

    # 1. Fundamental frequency (F0) — fast FFT-based method (~0.005s vs pyin ~60s)
    hop_len   = 512
    n_fft     = 2048
    S         = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_len))
    freqs     = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask      = (freqs >= 80) & (freqs <= 2000)
    S_masked  = S[mask]
    rms_frame = librosa.feature.rms(y=y, hop_length=hop_len)[0]
    energy_thresh = np.mean(rms_frame) * 0.3

    f0_frames, voiced_list = [], []
    for i in range(S_masked.shape[1]):
        if rms_frame[i] > energy_thresh:
            peak_bin = int(np.argmax(S_masked[:, i]))
            f0_frames.append(float(freqs[mask][peak_bin]))
            voiced_list.append(True)
        else:
            voiced_list.append(False)

    f0_valid = np.array(f0_frames) if f0_frames else np.array([300.0])
    features["f0_mean"]      = float(np.mean(f0_valid))
    features["f0_std"]       = float(np.std(f0_valid))
    features["f0_max"]       = float(np.max(f0_valid))
    features["f0_min"]       = float(np.min(f0_valid))
    features["f0_range"]     = float(np.max(f0_valid) - np.min(f0_valid))
    features["voiced_ratio"] = float(np.mean(voiced_list)) if voiced_list else 0.5
    features["f0_variation"] = float(np.std(np.diff(f0_valid))) if len(f0_valid) > 1 else 0.0

    # 2. MFCCs (timbre / vocal tract shape)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i}_std"]  = float(np.std(mfccs[i]))

    # 3. Spectral features
    spec_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    spec_flatness  = librosa.feature.spectral_flatness(y=y)[0]
    spec_contrast  = librosa.feature.spectral_contrast(y=y, sr=sr)

    features["spectral_centroid_mean"] = float(np.mean(spec_centroids))
    features["spectral_centroid_std"]  = float(np.std(spec_centroids))
    features["spectral_bandwidth_mean"]= float(np.mean(spec_bandwidth))
    features["spectral_rolloff_mean"]  = float(np.mean(spec_rolloff))
    features["spectral_flatness_mean"] = float(np.mean(spec_flatness))
    features["spectral_contrast_mean"] = float(np.mean(spec_contrast))

    # 4. Energy & amplitude dynamics
    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"]     = float(np.mean(rms))
    features["rms_std"]      = float(np.std(rms))
    features["rms_max"]      = float(np.max(rms))
    features["dynamic_range"]= float(np.max(rms) - np.min(rms))

    # 5. Rhythm / temporal structure
    duration = len(y) / sr
    features["duration"] = float(duration)

    # Crying segments (energy above threshold)
    threshold = np.mean(rms) * 0.5
    crying_frames = np.sum(rms > threshold)
    features["cry_ratio"] = float(crying_frames / len(rms)) if len(rms) > 0 else 0.0

    # Inter-cry pause analysis
    silent_frames = rms < (np.mean(rms) * 0.3)
    transitions = np.diff(silent_frames.astype(int))
    pause_count = int(np.sum(transitions == 1))
    features["pause_count"] = pause_count
    features["pause_rate"]  = float(pause_count / max(duration, 1))

    # Cry bout regularity (std of inter-peak intervals)
    peaks, _ = find_peaks(rms, height=np.mean(rms), distance=sr // 512 // 2)
    if len(peaks) > 2:
        intervals = np.diff(peaks)
        features["cry_regularity"] = float(1.0 / (np.std(intervals) / np.mean(intervals) + 1e-6))
    else:
        features["cry_regularity"] = 0.0

    # 6. Zero crossing rate (breathiness / noise)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"]  = float(np.std(zcr))

    # 7. Chroma (harmonic content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = float(np.mean(chroma))
    features["chroma_std"]  = float(np.std(chroma))

    # 8. Onset strength (abruptness of cry start)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features["onset_mean"] = float(np.mean(onset_env))
    features["onset_max"]  = float(np.max(onset_env))

    # 9. Mel spectrogram stats (overall spectral shape)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features["mel_mean"] = float(np.mean(mel_db))
    features["mel_std"]  = float(np.std(mel_db))

    # 10. Harmonic-to-noise ratio proxy
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy   = float(np.mean(y_harmonic ** 2))
    percussive_energy = float(np.mean(y_percussive ** 2))
    features["hnr"] = float(harmonic_energy / (percussive_energy + 1e-10))

    return features


def generate_waveform_plot(y, sr, archive_id):
    """Generate a beautiful waveform plot and save it."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')
    
    # Plot waveform
    time = np.linspace(0, len(y)/sr, len(y))
    ax.plot(time, y, color='#4f46e5', linewidth=1.5, alpha=0.8)
    
    # Styling
    ax.set_title('婴儿哭声波形图', fontsize=16, color='white', fontweight='bold', pad=20)
    ax.set_xlabel('时间 (秒)', fontsize=12, color='white')
    ax.set_ylabel('振幅', fontsize=12, color='white')
    ax.grid(True, alpha=0.3, color='white')
    ax.tick_params(colors='white')
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(ARCHIVES_DIR, f"{archive_id}_waveform.png")
    plt.savefig(plot_path, bbox_inches='tight', facecolor='#1e293b')
    plt.close()
    
    return plot_path


def extract_basic_info(y, sr):
    """Extract basic audio information."""
    info = {}
    
    # Duration
    info["duration_sec"] = float(len(y) / sr)
    
    # Sample rate
    info["sample_rate"] = int(sr)
    
    # Fundamental frequency (pitch)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=600, sr=sr)
    f0 = f0[voiced_flag]
    if len(f0) > 0:
        info["pitch_mean_hz"] = float(np.mean(f0))
        info["pitch_std_hz"] = float(np.std(f0))
        info["pitch_min_hz"] = float(np.min(f0))
        info["pitch_max_hz"] = float(np.max(f0))
    else:
        info["pitch_mean_hz"] = 0.0
        info["pitch_std_hz"] = 0.0
        info["pitch_min_hz"] = 0.0
        info["pitch_max_hz"] = 0.0
    
    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    info["rms_mean"] = float(np.mean(rms))
    info["rms_max"] = float(np.max(rms))
    
    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    info["spectral_centroid_mean_hz"] = float(np.mean(centroid))
    
    # Zero crossing rate (noisiness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    info["zero_crossing_rate_mean"] = float(np.mean(zcr))
    
    return info


def save_archive(audio_path, plot_path, info, archive_id):
    """Save archive metadata."""
    metadata = {
        "archive_id": archive_id,
        "timestamp": datetime.now().isoformat(),
        "audio_file": os.path.basename(audio_path),
        "plot_file": os.path.basename(plot_path),
        "info": info
    }
    
    metadata_path = os.path.join(ARCHIVES_DIR, f"{archive_id}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return metadata_path

STATES = [
    "hunger",
    "pain",
    "discomfort",
    "sleepy",
    "colic",
    "neurological_alert",
    "respiratory_alert",
    "startled",
    "boredom",
    "healthy_fussing",
]

STATE_LABELS = {
    "hunger":              "饥饿",
    "pain":                "疼痛",
    "discomfort":          "不适 / 尿布",
    "sleepy":              "困倦",
    "colic":               "肠绞痛",
    "neurological_alert":  "⚠️ 神经系统预警",
    "respiratory_alert":   "⚠️ 呼吸系统预警",
    "startled":            "受惊 / 突然不适",
    "boredom":             "无聊 / 想抱抱",
    "healthy_fussing":     "轻度烦躁",
}

STATE_ADVICE = {
    "hunger": [
        "距离上次喂奶是否已超过2小时？",
        "尝试哺乳或喂配方奶",
        "观察婴儿是否出现觅食反射（转头张嘴寻找乳头）",
    ],
    "pain": [
        "检查全身皮肤是否有红肿、皮疹或外伤",
        "检查衣物内是否有异物（如松脱的线头缠绕手指）",
        "如哭声持续超过20分钟不缓解，建议就医",
    ],
    "discomfort": [
        "检查尿布是否需要更换",
        "检查衣物是否过紧或有褶皱",
        "检查室内温度是否适宜（新生儿适宜22–26°C）",
        "尝试换一个抱姿或轻轻拍背排气",
    ],
    "sleepy": [
        "营造安静、光线较暗的睡眠环境",
        "轻轻摇晃或哼唱以帮助入睡",
        "检查是否已超过正常清醒时间（新生儿通常每次清醒45–90分钟）",
    ],
    "colic": [
        "尝试'飞机抱'（让宝宝趴在前臂，轻轻摇摆）",
        "腹部顺时针轻柔按摩",
        "检查母乳妈妈是否食用了可能导致胀气的食物（豆类、十字花科蔬菜）",
        "若配方奶喂养，考虑咨询医生是否需要更换低敏配方",
        "肠绞痛通常在宝宝3–4个月后自然缓解",
    ],
    "neurological_alert": [
        "⚠️ 哭声声学特征（极高音调 / 节律异常）与神经系统压力相关",
        "请记录本次哭声并尽快咨询儿科医生",
        "观察是否伴随囟门隆起、嗜睡、喂养困难等症状",
        "这是预警信号，非确定诊断——请及时就医评估",
    ],
    "respiratory_alert": [
        "⚠️ 哭声特征提示可能存在呼吸相关不适",
        "观察呼吸频率是否异常加快（正常新生儿30–60次/分钟）",
        "观察是否有鼻翼扇动、肋间凹陷等呼吸困难体征",
        "如症状明显，建议立即就医",
    ],
    "startled": [
        "检查是否有突然的噪音或光线变化",
        "用轻柔稳定的声音安抚宝宝",
        "将宝宝裹紧（包裹法/swaddling）有助于减少惊跳反射",
    ],
    "boredom": [
        "尝试面对面交流、做鬼脸或说话",
        "更换环境或给予新的视觉刺激（色彩鲜艳的玩具）",
        "提供肢体接触，如抱抱或皮肤接触",
    ],
    "healthy_fussing": [
        "宝宝目前处于轻度烦躁状态，尝试基本安抚",
        "按需喂养、更换体位、轻拍背部",
        "如果安抚无效超过30分钟，重新检查其他可能原因",
    ],
}

def _sigmoid(x, center, steepness=1.0):
    """Smooth 0-1 membership function, replaces hard if/else boundaries."""
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

def _gauss(x, center, sigma):
    """Gaussian membership — peaks at center, falls off symmetrically."""
    return float(np.exp(-0.5 * ((x - center) / sigma) ** 2))

def classify_cry(features):
    """
    Fuzzy multi-feature scoring with softmax sharpening.
    """
    scores = {s: 0.0 for s in STATES}

    f0      = features.get("f0_mean", 300)
    f0_std  = features.get("f0_std", 50)
    f0_max  = features.get("f0_max", 500)
    f0_var  = features.get("f0_variation", 10)
    rms     = features.get("rms_mean", 0.05)
    rms_std = features.get("rms_std", 0.02)
    dur     = features.get("duration", 5)
    pauses  = features.get("pause_rate", 1.0)
    reg     = features.get("cry_regularity", 5.0)
    cry_r   = features.get("cry_ratio", 0.5)
    zcr     = features.get("zcr_mean", 0.1)
    hnr     = features.get("hnr", 1.0)
    onset   = features.get("onset_max", 5.0)
    sc      = features.get("spectral_centroid_mean", 1000)
    sf      = features.get("spectral_flatness_mean", 0.01)

    # ── HUNGER ────────────────────────────────────────────────────────
    scores["hunger"] = (
        _gauss(f0, 380, 80)   * 40 +
        _sigmoid(pauses, 0.4, 6) * 25 +
        _sigmoid(reg, 3.0, 2)    * 20 +
        _gauss(rms, 0.07, 0.04)  * 10 +
        _sigmoid(f0_var, 12, 1)  * 5
    )

    # ── PAIN ──────────────────────────────────────────────────────────
    scores["pain"] = (
        _sigmoid(f0, 500, 0.015)   * 35 +
        _sigmoid(onset, 8, 0.5)    * 25 +
        (1 - _sigmoid(pauses, 0.35, 8)) * 20 +
        _sigmoid(rms, 0.10, 40)    * 15 +
        _sigmoid(f0_std, 80, 0.03) * 5
    )

    # ── DISCOMFORT ────────────────────────────────────────────────────
    scores["discomfort"] = (
        _gauss(f0, 300, 90)          * 30 +
        _gauss(pauses, 0.55, 0.2)    * 25 +
        _sigmoid(rms_std, 0.025, 60) * 25 +
        _gauss(rms, 0.06, 0.04)      * 20
    )

    # ── SLEEPY ────────────────────────────────────────────────────────
    scores["sleepy"] = (
        (1 - _sigmoid(rms, 0.055, 60))   * 35 +
        (1 - _sigmoid(f0, 380, 0.012))   * 20 +
        _sigmoid(pauses, 0.5, 6)          * 20 +
        (1 - _sigmoid(cry_r, 0.55, 8))   * 15 +
        (1 - _sigmoid(dur, 18, 0.3))      * 10
    )

    # ── COLIC ─────────────────────────────────────────────────────────
    scores["colic"] = (
        _sigmoid(f0, 460, 0.012)          * 20 +
        _sigmoid(dur, 18, 0.3)            * 25 +
        _sigmoid(rms, 0.09, 30)           * 15 +
        (1 - _sigmoid(reg, 2.5, 1.5))     * 25 +
        (1 - _sigmoid(pauses, 0.30, 8))   * 15
    )

    # ── NEUROLOGICAL ALERT ────────────────────────────────────────────
    neuro_f0    = _sigmoid(f0, 750, 0.008) * 50
    neuro_fmax  = _sigmoid(f0_max, 1200, 0.003) * 30
    neuro_hnr   = (1 - _sigmoid(hnr, 1.5, 2)) * 20
    scores["neurological_alert"] = neuro_f0 + neuro_fmax + neuro_hnr

    # ── RESPIRATORY ALERT ─────────────────────────────────────────────
    scores["respiratory_alert"] = (
        _sigmoid(sf, 0.04, 80)    * 35 +
        _sigmoid(zcr, 0.13, 40)   * 30 +
        _sigmoid(sc, 1800, 0.002) * 20 +
        _sigmoid(rms_std, 0.04, 40) * 15
    )

    # ── STARTLED ──────────────────────────────────────────────────────
    scores["startled"] = (
        _sigmoid(onset, 11, 1)             * 45 +
        (1 - _sigmoid(dur, 12, 0.4))       * 30 +
        _sigmoid(rms, 0.11, 30) * (1 - _sigmoid(cry_r, 0.45, 8)) * 25
    )

    # ── BOREDOM ───────────────────────────────────────────────────────
    scores["boredom"] = (
        (1 - _sigmoid(rms, 0.038, 80))   * 40 +
        _gauss(pauses, 0.65, 0.2)         * 30 +
        (1 - _sigmoid(f0, 310, 0.015))    * 15 +
        (1 - _sigmoid(cry_r, 0.38, 8))    * 15
    )

    # ── HEALTHY FUSSING (soft catch-all) ──────────────────────────────
    scores["healthy_fussing"] = (
        _gauss(f0, 420, 120)      * 30 +
        _gauss(rms, 0.065, 0.035) * 30 +
        _gauss(pauses, 0.5, 0.3)  * 20 +
        20
    )

    # ── SOFTMAX WITH TEMPERATURE SHARPENING ───────────────────────────
    T = 0.35
    raw = np.array([scores[s] for s in STATES], dtype=np.float64)
    raw = np.clip(raw, 0, None)
    if raw.max() == 0:
        raw += 1.0
    log_p = raw / T
    log_p -= log_p.max()
    exp_p = np.exp(log_p)
    probs = exp_p / exp_p.sum()

    result = {s: float(probs[i]) for i, s in enumerate(STATES)}
    sorted_scores = sorted(result.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores


# ─────────────────────────────────────────────
# HELPER: confidence label
# ─────────────────────────────────────────────

def confidence_label(prob):
    if prob > 0.45:
        return "很可能", "#16a34a"
    elif prob > 0.25:
        return "可能", "#ca8a04"
    elif prob > 0.12:
        return "较低可能", "#ea580c"
    else:
        return "低", "#dc2626"


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@router.get("/cry/health")
async def health():
    return {"status": "ok", "service": "BabyCry Archiver"}

@router.post("/cry/record")
async def record_audio(audio: UploadFile = File(...)):
    """Handle recorded audio upload and create archive."""
    if not audio.filename:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No audio file provided"}
        )

    # Generate unique archive ID
    archive_id = str(uuid.uuid4())
    
    # Save audio file
    audio_path = os.path.join(ARCHIVES_DIR, f"{archive_id}_audio.wav")
    with open(audio_path, "wb") as f:
        content = await audio.read()
        if len(content) == 0:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "上传的文件为空，请选择有效的音频文件"}
            )
        f.write(content)

    try:
        # Validate file size (minimum 1KB)
        if os.path.getsize(audio_path) < 1024:
            os.remove(audio_path)
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "文件太小，请上传有效的音频文件（至少1KB）"}
            )
        
        # Try to load audio with better format support
        try:
            # First try with soundfile (for WAV, FLAC, OGG)
            data, sr_native = sf.read(audio_path, dtype="float32", always_2d=False)
        except Exception as sf_error:
            try:
                # Fallback to pydub for other formats (WebM, MP3, M4A, etc.)
                print(f"Soundfile failed ({sf_error}), trying pydub...")
                audio_segment = AudioSegment.from_file(audio_path)
                
                # Convert to numpy array
                samples = np.array(audio_segment.get_array_of_samples())
                if audio_segment.channels == 2:
                    samples = samples.reshape((-1, 2))
                    samples = samples.mean(axis=1)  # Convert stereo to mono
                
                # Normalize to float32 between -1 and 1
                samples = samples.astype(np.float32)
                if audio_segment.sample_width == 2:  # 16-bit
                    samples /= 32768.0
                elif audio_segment.sample_width == 3:  # 24-bit
                    samples /= 8388608.0
                elif audio_segment.sample_width == 4:  # 32-bit
                    samples /= 2147483648.0
                
                data = samples
                sr_native = audio_segment.frame_rate
                
            except Exception as pydub_error:
                # Try scipy.io.wavfile as last resort
                try:
                    from scipy.io import wavfile
                    sr_native, data = wavfile.read(audio_path)
                    if data.dtype != np.float32:
                        if np.issubdtype(data.dtype, np.integer):
                            # Convert integer to float
                            if data.dtype == np.int16:
                                data = data.astype(np.float32) / 32768.0
                            elif data.dtype == np.int32:
                                data = data.astype(np.float32) / 2147483648.0
                        else:
                            data = data.astype(np.float32)
                    
                    if len(data.shape) > 1:
                        data = data.mean(axis=1)  # Convert to mono
                        
                except Exception as scipy_error:
                    os.remove(audio_path)
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "error": f"无法识别音频格式。支持的格式：WAV、FLAC、OGG、MP3、WebM、M4A。请尝试上传文件而不是实时录音，或确保文件未损坏。错误详情：soundfile({str(sf_error)[:50]}...) / pydub({str(pydub_error)[:50]}...) / scipy({str(scipy_error)[:50]}...)"}
                    )
        
        # Check if audio data is valid
        if len(data) == 0:
            os.remove(audio_path)
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "音频文件不包含有效数据"}
            )
        
        if data.ndim > 1:
            data = data.mean(axis=1)  # stereo → mono
        y = data.astype("float32")
        sr = 22050
        
        # Resample if needed
        if sr_native != sr:
            y = librosa.resample(y, orig_sr=sr_native, target_sr=sr)

        if len(y) < sr * 0.5:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "录音太短，请录制至少1秒的声音"}
            )

        # Extract basic info
        info = extract_basic_info(y, sr)
        
        # Generate waveform plot
        plot_path = generate_waveform_plot(y, sr, archive_id)
        
        # Save archive metadata
        metadata_path = save_archive(audio_path, plot_path, info, archive_id)

        return {
            "success": True,
            "archive_id": archive_id,
            "info": info,
            "plot_url": f"/cry/archive/{archive_id}/plot",
            "audio_url": f"/cry/archive/{archive_id}/audio"
        }

    except Exception as e:
        # Clean up on error
        for path in [audio_path]:
            if os.path.exists(path):
                os.remove(path)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"处理失败: {str(e)}"}
        )

@router.get("/cry/archive/{archive_id}/plot")
async def get_plot(archive_id: str):
    """Serve waveform plot image."""
    plot_path = os.path.join(ARCHIVES_DIR, f"{archive_id}_waveform.png")
    if not os.path.exists(plot_path):
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Plot not found"}
        )
    return FileResponse(plot_path, media_type="image/png")

@router.get("/cry/archive/{archive_id}/audio")
async def get_audio(archive_id: str):
    """Serve audio file."""
    audio_path = os.path.join(ARCHIVES_DIR, f"{archive_id}_audio.wav")
    if not os.path.exists(audio_path):
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Audio not found"}
        )
    return FileResponse(audio_path, media_type="audio/wav")

@router.get("/cry/archives")
async def list_archives():
    """List all cry archives."""
    archives = []
    for filename in os.listdir(ARCHIVES_DIR):
        if filename.endswith("_metadata.json"):
            archive_id = filename.replace("_metadata.json", "")
            metadata_path = os.path.join(ARCHIVES_DIR, filename)
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                archives.append({
                    "archive_id": archive_id,
                    "timestamp": metadata.get("timestamp"),
                    "duration": metadata.get("info", {}).get("duration_sec", 0),
                    "pitch_mean": metadata.get("info", {}).get("pitch_mean_hz", 0)
                })
            except:
                continue
    
    # Sort by timestamp descending
    archives.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return {"archives": archives}