# 数据收集功能使用指南

## 概述

所有带宽估计算法（BBR, GCC, Cubic, HRCC, PCC, Copa, Copa+）现在都支持数据收集功能，用于模仿学习训练。

## 启用数据收集

### 方法 1：通过环境变量（推荐）

在运行 `demo.py` 之前，设置环境变量：

```bash
# 启用数据收集
export ENABLE_DATA_COLLECTION=true
export DATA_COLLECTION_OUTPUT_DIR=share/output/imitation_data

# 运行 demo.py
python3 demo.py
```

或者在一行中运行：

```bash
ENABLE_DATA_COLLECTION=true DATA_COLLECTION_OUTPUT_DIR=share/output/imitation_data python3 demo.py
```

### 方法 2：修改 .env 文件

创建或修改 `.env` 文件：

```bash
# .env
ARG_A="BBR"
ENABLE_DATA_COLLECTION=true
DATA_COLLECTION_OUTPUT_DIR=share/output/imitation_data
```

然后运行：

```bash
python3 demo.py
```

## 数据保存位置

运行完成后，数据会保存在：

```
share/output/imitation_data/{算法名}_imitation_data.json
```

例如：
- `share/output/imitation_data/BBR_imitation_data.json`
- `share/output/imitation_data/GCC_imitation_data.json`
- `share/output/imitation_data/Cubic_imitation_data.json`
- 等等

## 数据格式

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

## 验证数据收集

运行完成后，检查数据文件：

```bash
# 查看生成的文件
ls -lh share/output/imitation_data/

# 查看文件内容（前几行）
head -30 share/output/imitation_data/BBR_imitation_data.json
```

## 注意事项

1. **默认关闭**：如果不设置环境变量，数据收集功能默认关闭，不影响原有功能
2. **文件自动保存**：数据会在程序结束时自动保存（通过析构函数）
3. **文件自动递增**：如果文件已存在，会自动递增文件名（如 `BBR_imitation_data_1.json`）
4. **Docker 环境**：数据会保存在挂载的 `share` 目录中，可以从宿主机访问

## 完整示例

```bash
# 1. 启用数据收集
export ENABLE_DATA_COLLECTION=true
export DATA_COLLECTION_OUTPUT_DIR=share/output/imitation_data

# 2. 运行 demo.py
python3 demo.py

# 3. 等待运行完成（Docker 容器会自动启动和停止）

# 4. 检查收集的数据
ls -lh share/output/imitation_data/
cat share/output/imitation_data/BBR_imitation_data.json | head -50
```

## 禁用数据收集

不设置环境变量或设置为 `false`：

```bash
# 不设置环境变量（默认关闭）
python3 demo.py

# 或显式设置为 false
ENABLE_DATA_COLLECTION=false python3 demo.py
```

