import pyaudio
import numpy as np
import time
import sys
import struct
from sound_transfer_common import (
    SAMPLE_RATE, FREQ_START, FREQ_ACK, FREQ_END,
    generate_sine, detect_freq, modulate_bytes, PREAMBLE
)

# 配置
START_DURATION = 0.5          # 开始/确认信号持续时间
START_REPEAT = 3              # 开始信号重复次数
START_INTERVAL = 0.2          # 开始信号间隔
ACK_TIMEOUT = 5.0             # 等待确认超时（秒）
CHUNK_SIZE = int(SAMPLE_RATE * 0.1)  # 录音块大小


def send_file(file_path):
    # 读取文件
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    print(f"准备发送文件: {file_path} ({len(file_data)} bytes)")

    # 构建数据帧
    data_len = len(file_data)
    checksum = sum(file_data) & 0xFF
    frame = PREAMBLE + struct.pack('<I', data_len) + file_data + bytes([checksum])
    print(f"数据帧总大小: {len(frame)} bytes")

    # 调制为音频
    audio_data, duration = modulate_bytes(frame)
    print(f"调制完成，音频时长: {duration:.2f} 秒")

    # 初始化PyAudio
    p = pyaudio.PyAudio()

    # --- 发送开始信号 ---
    print("发送开始信号...")
    stream_out = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=SAMPLE_RATE,
                        output=True)

    for _ in range(START_REPEAT):
        start_wave = generate_sine(FREQ_START, START_DURATION)
        stream_out.write(start_wave.astype(np.float32).tobytes())
        time.sleep(START_INTERVAL)

    # --- 等待确认信号 ---
    print("等待客户端确认...")
    stream_in = p.open(format=pyaudio.paFloat32,
                       channels=1,
                       rate=SAMPLE_RATE,
                       input=True,
                       frames_per_buffer=CHUNK_SIZE)

    start_time = time.time()
    ack_detected = False
    while time.time() - start_time < ACK_TIMEOUT:
        data = stream_in.read(CHUNK_SIZE, exception_on_overflow=False)
        chunk = np.frombuffer(data, dtype=np.float32)
        if detect_freq(chunk, FREQ_ACK, SAMPLE_RATE, threshold=100.0):
            ack_detected = True
            print("收到确认信号，开始发送数据...")
            break

    stream_in.stop_stream()
    stream_in.close()

    if not ack_detected:
        print("超时未收到确认信号，传输终止。")
        stream_out.stop_stream()
        stream_out.close()
        p.terminate()
        return

    # --- 发送数据 ---
    # 先发送一小段静音（避免开头截断）
    time.sleep(0.2)
    stream_out.write(audio_data.astype(np.float32).tobytes())

    # 发送结束信号
    end_wave = generate_sine(FREQ_END, START_DURATION)
    stream_out.write(end_wave.astype(np.float32).tobytes())

    print("数据发送完成。")
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python server.py <文件路径>")
        sys.exit(1)
    send_file(sys.argv[1])
