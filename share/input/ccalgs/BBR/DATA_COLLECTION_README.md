# BBR 数据收集功能说明

## 概述

BBR 的 `BandwidthEstimator.py` 现在支持直接记录带宽估计器的输入输出数据，用于模仿学习训练。这个功能可以确保记录的输入输出与算法实际使用的数据完全一致。

## 使用方法

### 基本使用

```python
from BandwidthEstimator import Estimator

# 启用数据收集
estimator = Estimator(
    enable_data_collection=True,
    output_dir="share/output/imitation_data"
)

# 正常使用
estimator.report_states(stats)
bandwidth = estimator.get_estimated_bandwidth()

# 手动保存（可选，析构函数会自动保存）
estimator.save_data()
```

### 禁用数据收集（默认）

```python
# 不启用数据收集，不影响原有功能
estimator = Estimator()
```

## 记录的数据格式

每条记录包含：

```json
{
  "timestamp": 0,
  "input_features": {
    "total_packets": 31,
    "video_packets": 19,
    "total_bytes": 19995,
    "video_bytes": 18671,
        "min_rtt_ms": 15.5,
        "avg_rtt_ms": 20.2,
        "max_rtt_ms": 30,
        "loss_rate": 0.0,
    "time_window_ms": 214,
    "throughput_bps": 747476.64,
        "delay_gradient_ms_per_s": -3.04,
        "current_bandwidth_bps": 3000000
  },
  "output_bandwidth": 4488820,
  "algorithm": "BBR"
}
```

## 特征说明

### 数据包统计
- `total_packets`: 总数据包数量
- `video_packets`: 视频数据包数量（payloadType=125）
- `total_bytes`: 总字节数
- `video_bytes`: 视频数据包字节数

### RTT/延迟统计
- `min_rtt_ms`: 最小 RTT（ms），从数据包的 receive_timestamp - send_timestamp 计算
- `avg_rtt_ms`: 平均 RTT（ms）
- `max_rtt_ms`: 最大 RTT（ms）

### 网络状态
- `loss_rate`: 基于序列号计算的丢包率（0.0-1.0）
- `time_window_ms`: 时间窗口大小（ms）
- `throughput_bps`: 吞吐量（bps）
- `delay_gradient_ms_per_s`: 延迟梯度（ms/s）

### 当前带宽
- `current_bandwidth_bps`: 当前带宽估计（bps），所有算法通用

## 与日志解析方法的对比

### 优势

1. **数据一致性保证**：
   - 记录的是算法实际使用的 `packets_list`
   - 特征提取在算法计算之后，使用算法内部状态
   - 输出是 `get_estimated_bandwidth()` 的实际返回值

2. **特征通用性**：
   - 使用通用特征，适用于所有算法（BBR, GCC, Cubic, HRCC, PCC, Copa, Copa+）
   - 特征提取基于数据包信息，不依赖算法特定实现

3. **时间精确性**：
   - 不需要时间对齐
   - 输入输出完全对应

### 劣势

1. **需要修改代码**：需要在每个算法的 `BandwidthEstimator.py` 中添加代码
2. **需要重新运行**：需要重新运行实验才能收集数据

## 数据保存

- 默认保存到：`share/output/imitation_data/BBR_imitation_data.json`
- 如果文件已存在，会自动递增文件名（`BBR_imitation_data_1.json` 等）
- 析构函数会自动保存数据，无需手动调用 `save_data()`

## 注意事项

1. **性能影响**：数据收集会略微增加内存和计算开销，但影响很小
2. **数据量**：每次 `get_estimated_bandwidth()` 调用都会记录一条数据
3. **文件大小**：长时间运行可能产生较大的 JSON 文件

## 与日志解析方法的关系

两种方法可以同时使用：
- **日志解析方法**：用于从已有日志中提取数据（不需要重新运行）
- **直接记录方法**：用于收集新的、完全一致的数据（需要重新运行）

两种方法的数据格式相同，可以合并使用。

