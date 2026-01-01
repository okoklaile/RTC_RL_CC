# FARC 带宽估计器

基于 **Fast Actor and Not-So-Furious Critic** 模型的带宽估计算法。

## 模型介绍

FARC 是一个使用离线强化学习训练的带宽估计模型，在 ACM MMSys 2024 带宽估计挑战赛中获得第二名。

- 论文: [Offline Reinforcement Learning for Bandwidth Estimation in RTC](https://dl.acm.org/doi/10.1145/3625468.3652184)
- 原始仓库: [FARC GitHub](https://github.com/ekremcet/FARC)

## 依赖安装

```bash
# 安装 ONNX Runtime (支持 GPU 加速)
pip install onnxruntime-gpu

# 或者使用 CPU 版本
pip install onnxruntime
```

## 使用方法

### 1. 基本用法

```python
from BandwidthEstimator import Estimator

# 创建估计器实例
estimator = Estimator()

# 接收数据包并更新状态
packet_stats = {
    "send_time_ms": 1000,        # 发送时间戳(毫秒)
    "arrival_time_ms": 1050,     # 到达时间戳(毫秒)
    "payload_type": 125,         # 载荷类型
    "sequence_number": 100,      # 序列号
    "ssrc": 12345,              # 同步源标识符
    "padding_length": 0,        # 填充长度(字节)
    "header_length": 12,        # 头部长度(字节)
    "payload_size": 1200        # 载荷大小(字节)
}

# 报告数据包状态
estimator.report_states(packet_stats)

# 获取带宽估计值 (bps)
bandwidth = estimator.get_estimated_bandwidth()
print(f"估计带宽: {bandwidth/1e6:.2f} Mbps")
```

### 2. 接口说明

#### `Estimator(model_path=None, use_onnx=True, step_time=60)`

**参数:**
- `model_path`: 模型文件路径，默认使用当前目录下的 ONNX 模型
- `use_onnx`: 是否优先使用 ONNX 模型（推荐）
- `step_time`: 推理步长(毫秒)，默认 60ms

#### `report_states(stats: dict)`

接收并记录数据包信息。

**参数:**
- `stats`: 数据包统计信息字典，必须包含以下字段：
  - `send_time_ms`: 发送时间戳(毫秒)
  - `arrival_time_ms`: 到达时间戳(毫秒)
  - `payload_type`: 载荷类型
  - `sequence_number`: 序列号
  - `ssrc`: 同步源标识符
  - `padding_length`: 填充长度(字节)
  - `header_length`: 头部长度(字节)
  - `payload_size`: 载荷大小(字节)

#### `get_estimated_bandwidth() -> int`

获取当前的带宽估计值。

**返回值:**
- `bandwidth`: 带宽估计值(bps)

### 3. 完整示例

```python
from BandwidthEstimator import Estimator

# 初始化估计器
estimator = Estimator()

# 模拟接收数据包序列
current_time = 0
send_time = 0
sequence_number = 0

for i in range(100):
    # 构造数据包统计信息
    packet_stats = {
        "send_time_ms": send_time,
        "arrival_time_ms": current_time,
        "payload_type": 125,
        "sequence_number": sequence_number,
        "ssrc": 12345,
        "padding_length": 0,
        "header_length": 12,
        "payload_size": 1200,
    }
    
    # 报告数据包状态
    estimator.report_states(packet_stats)
    
    # 每隔60ms获取一次带宽估计
    if (i + 1) % 3 == 0:
        bandwidth = estimator.get_estimated_bandwidth()
        print(f"时刻 {current_time}ms: 带宽 = {bandwidth/1e6:.2f} Mbps")
    
    # 更新时间
    current_time += 20  # 假设每20ms发送一个包
    send_time += 20
    sequence_number += 1
```

## 模型文件

- **fast_and_furious_model.onnx**: ONNX 格式的模型文件

## 特性

- ✅ 使用 ONNX Runtime 推理（支持 GPU/CPU）
- ✅ 符合库的标准接口规范
- ✅ 维护隐藏状态，支持序列预测
- ✅ 自动特征提取和归一化
- ✅ 带宽预测范围限制（80kbps - 20Mbps）
- ✅ 轻量级实现，无需 PyTorch 依赖

## 技术细节

### 模型架构
- **输入**: 150维特征向量（15个特征 × 10个时间步）
- **隐藏状态**: 128维（LSTM风格）
- **输出**: 2维动作向量（取第一维作为带宽预测）

### 特征组成（每组10个时间步）
1. 接收速率 (Receiving rate)
2. 接收包数 (Number of received packets)
3. 接收字节数 (Received bytes)
4. 排队延迟 (Queuing delay)
5. 延迟 (Delay)
6. 最小延迟 (Minimum seen delay)
7. 延迟比率 (Delay ratio)
8. 延迟差值 (Delay difference)
9. 包间隔时间 (Interarrival time)
10. 抖动 (Jitter)
11. 丢包率 (Packet loss ratio)
12. 平均丢包数 (Average lost packets)
13. 视频包概率 (Video packets probability)
14. 音频包概率 (Audio packets probability)
15. 探测包概率 (Probing packets probability)

## 性能

- 推理延迟: < 1ms (ONNX Runtime + GPU)
- 内存占用: ~50MB
- 支持实时带宽估计
- 无需 PyTorch，依赖更轻量

## 许可证

本实现基于原始 FARC 项目，遵循其开源许可证。

## 引用

如果使用本代码，请引用原始论文：

```bibtex
@inproceedings{FARC,
    author = {Çetinkaya, Ekrem and Pehlivanoglu, Ahmet and Ayten, Ihsan U. and Yumakogullari, Basar and Ozgun, Mehmet E. and Erinc, Yigit K. and Deniz, Enes and Begen, Ali C.},
    booktitle = {Proceedings of the 15th ACM Multimedia Systems Conference},
    title = {{Offline Reinforcement Learning for Bandwidth Estimation in RTC Using a Fast Actor and Not-So-Furious Critic}},
    year = {2024}
}
```

