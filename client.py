import pyaudio
import numpy as np
import time
import sys
from sound_transfer_common import (
    SAMPLE_RATE, FREQ_START, FREQ_ACK, FREQ_END,
    generate_sine, detect_freq, demodulate_audio, ENERGY_THRESHOLD
)

# 配置
LISTEN_CHUNK = int(SAMPLE_RATE * 0.1)      # 监听块大小
START_REQUIRED_COUNT = 3                   # 需要连续检测到开始信号的次数
RECORD_TIMEOUT = 30.0                      # 最大录音时长（秒）


def receive_file(save_path):
    p = pyaudio.PyAudio()

    # --- 监听开始信号 ---
    print("监听开始信号...")
    stream_in = p.open(format=pyaudio.paFloat32,
                       channels=1,
                       rate=SAMPLE_RATE,
                       input=True,
                       frames_per_buffer=LISTEN_CHUNK)

    start_detected = 0
    while start_detected < START_REQUIRED_COUNT:
        data = stream_in.read(LISTEN_CHUNK, exception_on_overflow=False)
        chunk = np.frombuffer(data, dtype=np.float32)
        if detect_freq(chunk, FREQ_START, SAMPLE_RATE, ENERGY_THRESHOLD):
            start_detected += 1
        else:
            start_detected = 0  # 重置计数（要求连续检测）

    print("检测到开始信号，发送确认...")

    # --- 发送确认信号 ---
    stream_out = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=SAMPLE_RATE,
                        output=True)
    ack_wave = generate_sine(FREQ_ACK, 0.5)
    stream_out.write(ack_wave.astype(np.float32).tobytes())
    stream_out.stop_stream()
    stream_out.close()

    # --- 开始录音（接收数据）---
    print("开始接收数据...")
    # 重新打开输入流（从头开始录制）
    stream_in.stop_stream()
    stream_in.close()
    stream_in = p.open(format=pyaudio.paFloat32,
                       channels=1,
                       rate=SAMPLE_RATE,
                       input=True,
                       frames_per_buffer=LISTEN_CHUNK)

    recorded_frames = []
    start_time = time.time()
    end_detected = False

    while time.time() - start_time < RECORD_TIMEOUT and not end_detected:
        data = stream_in.read(LISTEN_CHUNK, exception_on_overflow=False)
        chunk = np.frombuffer(data, dtype=np.float32)
        recorded_frames.append(chunk)
        # 检测结束信号（提前停止录音）
        if detect_freq(chunk, FREQ_END, SAMPLE_RATE, ENERGY_THRESHOLD):
            end_detected = True
            print("检测到结束信号，停止录音。")

    stream_in.stop_stream()
    stream_in.close()
    p.terminate()

    # 合并录音数据
    audio_data = np.concatenate(recorded_frames)
    print(f"录音完成，时长: {len(audio_data)/SAMPLE_RATE:.2f} 秒")

    # 解调数据
    print("解调数据...")
    file_data = demodulate_audio(audio_data, SAMPLE_RATE)
    if file_data is None:
        print("解调失败：未找到有效前导码或校验错误。")
        return

    # 保存文件
    with open(save_path, 'wb') as f:
        f.write(file_data)
    print(f"文件已保存到: {save_path} (大小: {len(file_data)} bytes)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python client.py <保存文件路径>")
        sys.exit(1)
    receive_file(sys.argv[1])
