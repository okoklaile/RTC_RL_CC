# WebRTC 日志解析 - 可提取的输入输出字段说明

## 概述

`parse_webrtc_log.py` 脚本可以从 WebRTC 日志文件中解析出带宽估计器的输入输出数据，用于模仿学习训练。

## 输出字段（BWE Estimation）

从日志中的 `Send back BWE estimation: <value> at time: <timestamp>` 提取：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `output_bandwidth` | int | 带宽估计值（bps） |
| `timestamp` | int | BWE estimation 的时间戳（ms，相对时间） |

## 输入字段（从数据包信息提取的特征）

### 1. 数据包统计

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `total_packets` | int | 时间窗口内的总数据包数量 |
| `video_packets` | int | 视频数据包数量（payloadType=125） |
| `total_bytes` | int | 总字节数 |
| `video_bytes` | int | 视频数据包字节数 |

### 2. RTT/延迟统计

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `min_rtt_ms` | float | 最小数据包到达间隔（ms） |
| `avg_rtt_ms` | float | 平均数据包到达间隔（ms） |
| `max_rtt_ms` | float | 最大数据包到达间隔（ms） |

**注意**：由于日志中 `sendTimestamp` 和 `arrivalTimeMs` 使用不同的时间基准，无法直接计算真实的 RTT。这里使用数据包到达间隔作为延迟的近似。

### 3. 丢包率

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `loss_rate` | float | 平均丢包率（从 `lossRates` 字段提取，范围 0.0-1.0） |
| `packet_loss_rate` | float | 基于序列号计算的丢包率（范围 0.0-1.0） |

### 4. 吞吐量和时间窗口

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `time_window_ms` | int | 时间窗口大小（ms），即第一个和最后一个数据包的时间差 |
| `throughput_bps` | float | 吞吐量（bps），计算公式：`(total_bytes * 8 * 1000) / time_window_ms` |

### 5. 延迟梯度

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `delay_gradient_ms_per_s` | float | 延迟梯度（ms/s），表示数据包到达间隔的变化率。正值表示延迟增加，负值表示延迟减少 |

### 6. 当前带宽

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `current_bandwidth_bps` | float | 当前 pacer pacing rate（bps）。注意：如果值为 `1.7976931348623157e+308`，表示无效值（日志中的占位符） |

## 原始数据包字段（解析过程中使用）

以下字段在解析过程中被提取，但最终特征中已聚合：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `arrivalTimeMs` | int | 数据包到达时间（ms，绝对时间戳） |
| `sendTimestamp` | int | 数据包发送时间戳（相对时间，从会话开始） |
| `sequenceNumber` | int | RTP 序列号 |
| `ssrc` | int | 同步源标识符 |
| `payloadType` | int | 负载类型（111=音频，125=视频） |
| `headerLength` | int | RTP 头部长度（字节） |
| `paddingLength` | int | 填充长度（字节） |
| `payloadSize` | int | 负载大小（字节） |
| `lossRates` | float | 丢包率（0.0-1.0） |
| `pacerPacingRate` | float | Pacer 发送速率（bps） |
| `pacerPaddingRate` | float | Pacer 填充速率（bps） |
| `hasTransportSequenceNumber` | bool | 是否有传输序列号 |

## 数据格式

解析后的数据保存为 JSON 格式：

```json
{
  "algorithm": "BBR",
  "total_records": 290,
  "data": [
    {
      "timestamp": 1202700425,
      "input_features": {
        "total_packets": 31,
        "video_packets": 19,
        "total_bytes": 19995,
        "video_bytes": 18671,
        "min_rtt_ms": 0,
        "avg_rtt_ms": 5.47,
        "max_rtt_ms": 30,
        "loss_rate": 0.0,
        "packet_loss_rate": 0.0,
        "time_window_ms": 214,
        "throughput_bps": 747476.64,
        "delay_gradient_ms_per_s": -3.04,
        "current_bandwidth_bps": 4488819.0
      },
      "output_bandwidth": 4488820,
      "algorithm": "BBR"
    }
  ]
}
```

## 使用方法

```bash
# 解析日志文件
python3 utils/parse_webrtc_log.py share/output/trace/webrtc_BBR.log

# 指定输出文件
python3 utils/parse_webrtc_log.py share/output/trace/webrtc_BBR.log output.json

# 查看可提取字段说明
python3 utils/parse_webrtc_log.py --help-fields
```

## 注意事项

1. **时间对齐**：BWE estimation 的时间戳和数据包时间戳使用不同的基准，脚本会自动计算时间偏移量进行对齐。

2. **时间窗口**：默认使用每个 BWE estimation 之前最近的 50 个数据包来提取特征。

3. **无效值处理**：某些字段可能包含无效值（如 `1.7976931348623157e+308`），在使用数据时需要进行过滤。

4. **RTT 限制**：由于时间基准不同，无法直接计算真实的 RTT，使用数据包到达间隔作为近似。

5. **特征完整性**：某些特征（如吞吐量、延迟梯度）需要至少 2-4 个数据包才能计算，如果数据包不足，会使用默认值 0。

## 与 HRCC 输入特征的对比

HRCC 使用以下 6 个特征：
1. `receiving_rate` - 接收速率
2. `delay` - 延迟
3. `loss_ratio` - 丢包率
4. `bandwidth_prediction` - 带宽预测
5. `overuse_distance` - 过载距离
6. `last_overuse_cap` - 上次过载容量

从日志中提取的特征可以映射到：
- `receiving_rate` ← `throughput_bps`
- `delay` ← `avg_rtt_ms`
- `loss_ratio` ← `loss_rate` 或 `packet_loss_rate`
- `bandwidth_prediction` ← `current_bandwidth_bps`
- `overuse_distance` ← `delay_gradient_ms_per_s`（需要转换）
- `last_overuse_cap` ← 需要额外计算（日志中可能没有）

