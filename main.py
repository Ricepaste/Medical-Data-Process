import json
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram, iirnotch
from datetime import datetime

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    設計一個帶通濾波器。
    
    參數：
    lowcut (float): 低頻截止頻率。
    highcut (float): 高頻截止頻率。
    fs (float): 取樣頻率。
    order (int): 濾波器的階數，預設為5。
    
    回傳：
    b, a (tuple): 濾波器的係數。
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    使用帶通濾波器對數據進行濾波。
    
    參數：
    data (array): 要處理的訊號。
    lowcut (float): 低頻截止頻率。
    highcut (float): 高頻截止頻率。
    fs (float): 取樣頻率。
    order (int): 濾波器的階數，預設為5。
    
    回傳：
    array: 濾波後的訊號。
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)  # 零相位濾波

def butter_notch(notch_freq, quality_factor, fs):
    """
    設計一個陷波濾波器來去除特定頻率的干擾，例如電源頻率 (60Hz)。
    
    參數：
    notch_freq (float): 陷波頻率 (Hz)，通常為電源干擾頻率，例如 50Hz 或 60Hz。
    quality_factor (float): 品質因子，決定濾波器的帶寬，較高的值代表較窄的帶寬。
    fs (float): 取樣頻率。
    
    回傳：
    b, a (tuple): 濾波器的係數。
    """
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = iirnotch(freq, quality_factor, fs)
    return b, a

def butter_notch_filter(data, notch_freq, quality_factor, fs):
    """
    使用陷波濾波器來去除特定頻率的干擾。
    
    參數：
    data (array): 要處理的訊號。
    notch_freq (float): 陷波頻率 (Hz)。
    quality_factor (float): 品質因子，決定濾波器的帶寬。
    fs (float): 取樣頻率。
    
    回傳：
    array: 濾波後的訊號。
    """
    b, a = butter_notch(notch_freq, quality_factor, fs)
    return filtfilt(b, a, data)

def wavelet_denoise(data, wavelet='db4', level=3, threshold_factor=2.0):
    """
    使用小波轉換進行去雜訊處理。
    
    參數：
    data (array): 要處理的訊號。
    wavelet (str): 小波函數，預設為 'db4'。
    level (int): 分解層數，影響頻率解析度。
    threshold_factor (float): 閾值係數，影響去雜訊程度。
    
    回傳：
    array: 去雜訊後的訊號。
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    threshold = threshold_factor * np.std(coeffs[-1])
    coeffs = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

def snr(signal, noise):
    """
    計算訊號與雜訊的信噪比 (SNR)。
    
    參數：
    signal (array): 原始訊號。
    noise (array): 雜訊訊號。
    
    回傳：
    float: 訊噪比 (dB)。
    """
    power_signal = np.mean(signal ** 2)
    power_noise = np.mean(noise ** 2)
    return 10 * np.log10(power_signal / power_noise)

def plot_signal(time_axis, original, filtered, denoised, title_prefix, filename):
    """
    繪製原始訊號、濾波後訊號與去雜訊訊號。
    
    參數：
    time_axis (array): 時間軸。
    original (array): 原始訊號。
    filtered (array): 濾波後訊號。
    denoised (array): 去雜訊後訊號。
    title_prefix (str): 圖片標題前綴。
    filename (str): 儲存圖片的檔名。
    """
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, original, 'b', label='Original Signal')
    plt.legend(loc='upper right')
    plt.title(f'{title_prefix} - Original Signal')
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, filtered, 'g', label='Filtered Signal')
    plt.legend(loc='upper right')
    plt.title(f'{title_prefix} - Filtered Signal')
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, denoised, 'r', label='Denoised Signal')
    plt.legend(loc='upper right')
    plt.title(f'{title_prefix} - Denoised Signal')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def time_frequency_analysis(signal, fs, title, filename):
    """
    進行時間-頻率分析，計算並繪製信號的頻譜圖。
    
    參數:
    signal (numpy.ndarray): 需要分析的輸入信號。
    fs (int): 取樣率 (Hz)，用於時間軸和頻率軸的計算。
    title (str): 圖表的標題。
    filename (str): 圖片儲存的檔案名稱。
    """
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=fs//2, noverlap=fs//4)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')  # 避免 log(0) 問題
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'Spectrogram - {title}')
    plt.colorbar(label='Power/Frequency [dB/Hz]')
    plt.savefig(filename)
    plt.close()


def process_json(json_file, channels, fs, lowcut, highcut, order=5):
    """
    處理 JSON 格式的 EEG 數據，進行預處理與濾波，並分析時頻特性。
    
    參數:
    json_file (str): 輸入的 JSON 檔案路徑。
    channels (list of str): 需要處理的通道名稱。
    fs (int): 取樣率 (Hz)。
    lowcut (float): 頻帶通濾波的低頻界限 (Hz)。
    highcut (float): 頻帶通濾波的高頻界限 (Hz)。
    order (int, 可選): 濾波器的階數，預設為 5。
    """
    data = {ch: [] for ch in channels}
    timestamps = []
    
    with open(json_file, 'r') as f:
        timestamps = []
        for line in f:
            try:
                record = json.loads(line)
                # 解析時間戳並轉換為 Unix 時間 (秒)
                dt = datetime.strptime(record['timeStamp'], "%Y-%m-%d %H-%M-%S-%f")
                timestamps.append(dt.timestamp())
                
                for ch in channels:
                    data[ch].append(float(record[ch]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

        timestamps = np.array(timestamps)
        time_axis = timestamps - timestamps[0]  # 轉換為從 0 秒開始的時間軸

    for ch in channels:
        raw_signal = np.array(data[ch])
        
        # 進行 60Hz 陷波濾波，消除電源干擾 (品質因子設為 30.0)
        notched_signal = butter_notch_filter(raw_signal, 60.0, 30.0, fs)
        
        # 進行帶通濾波，保留 8-30Hz 頻段 (Alpha + Beta 波段)
        filtered_signal = butter_bandpass_filter(notched_signal, lowcut, highcut, fs, order)
        
        # 使用小波變換進行去雜訊
        denoised_signal = wavelet_denoise(filtered_signal)
        
        # 計算去雜訊前後的差異 (雜訊成分)
        filtered_diff = filtered_signal - denoised_signal
        
        print(f'{ch} SNR: {snr(raw_signal, filtered_signal):.7f} dB (Raw vs. Filtered)')
        print(f'{ch} SNR: {snr(filtered_signal, denoised_signal):.7f} dB (Filtered vs. Denoised)')
        print(f'{ch} SNR: {snr(raw_signal, denoised_signal):.7f} dB (Raw vs. Denoised)')
        
        # 繪製雜訊信號
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, filtered_diff, 'b', label='Noise Signal')
        plt.legend(loc='upper right')
        plt.title(f'{ch} - Noise Signal')
        plt.savefig(f'{ch}_noise_signal.png')
        plt.close()

        # 繪製原始、濾波後及去雜訊後的信號圖
        plot_signal(time_axis, raw_signal, filtered_signal, denoised_signal, f'{ch} Analysis', f'{ch}_analysis.png')
        
        # 進行時頻分析並儲存頻譜圖
        time_frequency_analysis(raw_signal, fs, f'{ch} Raw', f'{ch}_raw_spectrogram.png')
        time_frequency_analysis(filtered_signal, fs, f'{ch} Filtered', f'{ch}_filtered_spectrogram.png')
        time_frequency_analysis(denoised_signal, fs, f'{ch} Denoised', f'{ch}_denoised_spectrogram.png')

if __name__ == "__main__":
    # json_file1 = 'data/concussionganglion_vrca_13_EegSsvep.json'
    json_file2 = 'data/concussionbrainbit_vrca_13_EegSsvep.json'
    fs = 250  # 根據 OpenBCI/BrainBit 設定適當的取樣率
    lowcut, highcut = 8.0, 30.0  # 適合視覺相關的 Alpha + Beta 頻段

    
    # process_json(json_file1, ['ch4'], fs, lowcut, highcut)  # OpenBCI Ganglion OZ 點位
    process_json(json_file2, ['ch3', 'ch4'], fs, lowcut, highcut)  # BrainBit ch3, ch4
