import numpy as np
import struct

# 音频参数
SAMPLE_RATE = 44100
BIT_DURATION = 0.1                # 每个比特持续时间（秒）
SAMPLES_PER_BIT = int(SAMPLE_RATE * BIT_DURATION)

# 频率定义
FREQ_0 = 1000      # 比特0
FREQ_1 = 2000      # 比特1
FREQ_START = 3000  # 开始信号
FREQ_ACK = 1500    # 确认信号
FREQ_END = 3500    # 结束信号

# 检测阈值（幅度平方，需根据实际环境调整）
ENERGY_THRESHOLD = 100.0

# 前导码
PREAMBLE = bytes([0xAA, 0x55, 0x00, 0xFF])


def generate_sine(freq, duration, sample_rate=SAMPLE_RATE):
    """生成指定频率和持续时间的正弦波（-1~1）"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    return wave.astype(np.float32)


def goertzel(samples, target_freq, sample_rate):
    """
    Goertzel算法计算特定频率的能量（幅度的平方）
    """
    n = len(samples)
    k = int(0.5 + n * target_freq / sample_rate)
    if k >= n:
        k = n - 1
    omega = 2.0 * np.pi * k / n
    coeff = 2.0 * np.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for sample in samples:
        s = sample + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
    return power


def detect_freq(audio_chunk, target_freq, sample_rate, threshold):
    """检测音频块中是否包含目标频率（能量超过阈值）"""
    energy = goertzel(audio_chunk, target_freq, sample_rate)
    return energy > threshold


def modulate_bytes(data_bytes):
    """
    将字节数据调制成音频波形（FSK）
    返回: (audio_data, duration_seconds)
    """
    # 将字节转换为比特列表（LSB first）
    bits = []
    for b in data_bytes:
        for i in range(8):
            bits.append((b >> i) & 1)
    
    # 为每个比特生成对应的正弦波
    audio_parts = []
    for bit in bits:
        freq = FREQ_1 if bit else FREQ_0
        wave = generate_sine(freq, BIT_DURATION)
        audio_parts.append(wave)
    
    audio_data = np.concatenate(audio_parts)
    duration = len(bits) * BIT_DURATION
    return audio_data, duration


def demodulate_audio(audio_data, sample_rate):
    """
    解调音频数据，恢复原始字节（需要前导码同步）
    返回: bytes 或 None（未找到前导码）
    """
    total_samples = len(audio_data)
    num_bits = total_samples // SAMPLES_PER_BIT
    if num_bits == 0:
        return None
    
    # 逐比特解调
    bit_values = []
    for i in range(num_bits):
        start = i * SAMPLES_PER_BIT
        end = start + SAMPLES_PER_BIT
        chunk = audio_data[start:end]
        e0 = goertzel(chunk, FREQ_0, sample_rate)
        e1 = goertzel(chunk, FREQ_1, sample_rate)
        bit = 1 if e1 > e0 else 0
        bit_values.append(bit)
    
    # 将比特流转换为字节（每8位一个字节，LSB first）
    byte_data = bytearray()
    for i in range(0, len(bit_values) - 7, 8):
        byte = 0
        for j in range(8):
            byte |= (bit_values[i + j] << j)
        byte_data.append(byte)
    
    # 搜索前导码
    preamble_len = len(PREAMBLE)
    for i in range(len(byte_data) - preamble_len):
        if byte_data[i:i+preamble_len] == PREAMBLE:
            # 找到前导码，解析数据长度
            offset = i + preamble_len
            if offset + 4 > len(byte_data):
                return None
            data_len = struct.unpack('<I', byte_data[offset:offset+4])[0]
            offset += 4
            if offset + data_len + 1 > len(byte_data):
                return None
            file_data = byte_data[offset:offset+data_len]
            offset += data_len
            received_checksum = byte_data[offset]
            # 校验和计算（累加和模256）
            calc_checksum = sum(file_data) & 0xFF
            if calc_checksum == received_checksum:
                return bytes(file_data)
            else:
                return None
    return None
