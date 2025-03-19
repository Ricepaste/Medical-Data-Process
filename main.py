import json
import numpy as np
from scipy.signal import butter, lfilter, spectrogram

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Butterworth 帶通濾波器設計 (同前)。
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    應用 Butterworth 帶通濾波器 (同前)。
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def time_frequency_analysis(signal, fs, channel_name, filename_prefix):
    """
    執行時頻分析並儲存 spectrogram 圖片。

    Args:
        signal (np.array): 要分析的訊號資料
        fs (float): 取樣率 (Hz)
        channel_name (str): 通道名稱 (例如 'ch3')，用於圖表標題和檔名
        filename_prefix (str): 檔案名稱前綴，用於區分原始和濾波後訊號
    """
    f, t, Sxx = spectrogram(signal, fs=fs)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(
        t, f, 10 * np.log10(Sxx), shading="gouraud"
    )  # 顯示功率譜密度 (PSD) in dB
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.title(f"{filename_prefix} {channel_name} Spectrogram")
    plt.colorbar(label="Power/Frequency [dB/Hz]")  # 顏色條標籤
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{channel_name}_BrainBit_spectrogram.png")
    plt.close()  # 關閉圖形，避免顯示


def filter_eeg_data_and_plot(json_file_path, lowcut, highcut, fs, order=5):
    """
    讀取 JSON EEG 資料，對 ch3 和 ch4 進行帶通濾波，繪製並儲存原始訊號、濾波後訊號和 spectrogram 圖片。

    Args:
        json_file_path (str): JSON 檔案路徑
        lowcut (float): 低頻截止頻率 (Hz)
        highcut (float): 高頻截止頻率 (Hz)
        fs (float):  取樣率 (Hz)
        order (int): 濾波器階數
    """
    original_ch3_values = []
    original_ch4_values = []
    filtered_ch3_values = []
    filtered_ch4_values = []
    timestamps = []

    with open(json_file_path, "r") as f:
        for line in f:
            try:
                eeg_data = json.loads(line)

                if "ch3" in eeg_data and "ch4" in eeg_data and "timeStamp" in eeg_data:
                    try:
                        ch3_value = float(eeg_data["ch3"])
                        ch4_value = float(eeg_data["ch4"])
                        timestamp = eeg_data["timeStamp"]

                        original_ch3_values.append(ch3_value)
                        original_ch4_values.append(ch4_value)
                        timestamps.append(timestamp)

                        ch3_data = np.array([ch3_value])
                        ch4_data = np.array([ch4_value])

                        filtered_ch3 = butter_bandpass_filter(
                            ch3_data, lowcut, highcut, fs, order
                        )[0]
                        filtered_ch4 = butter_bandpass_filter(
                            ch4_data, lowcut, highcut, fs, order
                        )[0]

                        filtered_ch3_values.append(filtered_ch3)
                        filtered_ch4_values.append(filtered_ch4)

                        eeg_data["ch3"] = str(filtered_ch3)
                        eeg_data["ch4"] = str(filtered_ch4)

                    except ValueError:
                        print(
                            f"警告: ch3 或 ch4 的值無法轉換為數值，跳過濾波，原始資料為: {eeg_data}"
                        )
                else:
                    print(f"警告: JSON 資料缺少 ch3, ch4 或 timeStamp 欄位: {eeg_data}")

            except json.JSONDecodeError:
                print(f"警告: 無法解析 JSON 行: {line.strip()}")

    time_axis = np.arange(len(timestamps))

    # 繪製並儲存原始訊號圖
    fig_original, axs_original = plt.subplots(2, 1, figsize=(12, 6))
    axs_original[0].plot(time_axis, original_ch3_values)
    axs_original[0].set_title("Original Ch3")
    axs_original[0].set_xlabel("Data Point Index")
    axs_original[0].set_ylabel("Amplitude")
    axs_original[0].grid(True)

    axs_original[1].plot(time_axis, original_ch4_values)
    axs_original[1].set_title("Original Ch4")
    axs_original[1].set_xlabel("Data Point Index")
    axs_original[1].set_ylabel("Amplitude")
    axs_original[1].grid(True)

    plt.tight_layout()
    plt.savefig("original_BrainBit_signal_plots.png")
    plt.close(fig_original)  # 關閉原始訊號圖

    # 繪製並儲存濾波後訊號圖
    fig_filtered, axs_filtered = plt.subplots(2, 1, figsize=(12, 6))
    axs_filtered[0].plot(time_axis, filtered_ch3_values)
    axs_filtered[0].set_title("Filtered Ch3")
    axs_filtered[0].set_xlabel("Data Point Index")
    axs_filtered[0].set_ylabel("Amplitude")
    axs_filtered[0].grid(True)

    axs_filtered[1].plot(time_axis, filtered_ch4_values)
    axs_filtered[1].set_title("Filtered Ch4")
    axs_filtered[1].set_xlabel("Data Point Index")
    axs_filtered[1].set_ylabel("Amplitude")
    axs_filtered[1].grid(True)

    plt.tight_layout()
    plt.savefig("filtered_BrainBit_signal_plots.png")
    plt.close(fig_filtered)  # 關閉濾波後訊號圖

    # 進行時頻分析並儲存 spectrogram 圖片
    original_ch3_array = np.array(
        original_ch3_values
    )  # Convert lists to numpy arrays for spectrogram function
    original_ch4_array = np.array(original_ch4_values)
    filtered_ch3_array = np.array(filtered_ch3_values)
    filtered_ch4_array = np.array(filtered_ch4_values)

    time_frequency_analysis(original_ch3_array, fs, "Ch3", "original")
    time_frequency_analysis(original_ch4_array, fs, "Ch4", "original")
    time_frequency_analysis(filtered_ch3_array, fs, "Ch3", "filtered")
    time_frequency_analysis(filtered_ch4_array, fs, "Ch4", "filtered")


if __name__ == "__main__":
    json_file = (
        "data/concussionganglion_vrca_9_EegSsvep.json"  # 替換成您的 JSON 檔案路徑
    )
    sampling_rate = 128.0
    low_frequency = 1.0
    high_frequency = 40.0
    filter_order = 5

    filter_eeg_data_and_plot(
        json_file, low_frequency, high_frequency, sampling_rate, filter_order
    )
    print(
        "圖片已儲存為 original_signal_plots.png, filtered_signal_plots.png, 以及 spectrogram 圖片。"
    )
