#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 webrtc log 文件中解析带宽估计器的输入输出
提取 BWE estimation 和对应的数据包信息，用于模仿学习数据收集
"""

import re
import json
import os
import sys
from typing import List, Dict, Optional, Tuple
from collections import deque
from pathlib import Path


class WebRTCLogParser:
    """WebRTC 日志解析器"""
    
    def __init__(self, log_path: str, window_size: int = 50):
        """
        初始化解析器
        
        Args:
            log_path: webrtc log 文件路径
            window_size: 用于特征提取的时间窗口大小（数据包数量）
        """
        self.log_path = log_path
        self.window_size = window_size
        self.bwe_estimations = []  # [(timestamp, bwe_value)]
        self.packet_infos = []  # [packet_dict]
        
    def parse(self) -> List[Dict]:
        """
        解析日志文件
        
        Returns:
            解析后的记录列表，每条记录包含输入特征和输出带宽
        """
        print(f"正在解析日志文件: {self.log_path}")
        
        # 第一步：提取所有 BWE estimations 和数据包信息
        self._extract_all_data()
        
        print(f"  提取到 {len(self.bwe_estimations)} 个 BWE estimations")
        print(f"  提取到 {len(self.packet_infos)} 个数据包信息")
        
        # 第二步：按时间对齐并生成记录
        records = self._align_and_extract_features()
        
        print(f"  生成 {len(records)} 条完整记录")
        return records
    
    def _extract_all_data(self):
        """提取所有 BWE estimations 和数据包信息"""
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 提取 BWE estimation
                bwe_match = re.search(
                    r'Send back BWE estimation:\s*([\d\.eE\+\-]+)\s*at time:\s*(\d+)',
                    line
                )
                if bwe_match:
                    bwe_value = float(bwe_match.group(1))
                    bwe_time = int(bwe_match.group(2))
                    self.bwe_estimations.append((bwe_time, bwe_value))
                
                # 提取数据包信息
                if 'remote_estimator_proxy.cc:155' in line:
                    packet_info = self._parse_packet_info(line)
                    if packet_info:
                        self.packet_infos.append(packet_info)
    
    def _parse_packet_info(self, line: str) -> Optional[Dict]:
        """解析单行数据包信息"""
        try:
            json_start = line.find('{')
            if json_start == -1:
                return None
            
            json_str = line[json_start:]
            packet_data = json.loads(json_str)
            
            packet_info = packet_data.get('packetInfo', {})
            header = packet_info.get('header', {})
            
            # 提取所有可用字段
            parsed = {
                # 时间信息
                'arrivalTimeMs': packet_info.get('arrivalTimeMs', 0),
                'sendTimestamp': header.get('sendTimestamp', 0),
                
                # 数据包标识
                'sequenceNumber': header.get('sequenceNumber', 0),
                'ssrc': header.get('ssrc', 0),
                'payloadType': header.get('payloadType', 0),
                
                # 数据包大小
                'headerLength': header.get('headerLength', 0),
                'paddingLength': header.get('paddingLength', 0),
                'payloadSize': packet_info.get('payloadSize', 0),
                'totalSize': (
                    header.get('headerLength', 0) +
                    packet_info.get('payloadSize', 0) +
                    header.get('paddingLength', 0)
                ),
                
                # 网络状态
                'lossRates': packet_info.get('lossRates', 0.0),
                
                # Pacer 信息
                'pacerPacingRate': packet_data.get('pacerPacingRate', 0),
                'pacerPaddingRate': packet_data.get('pacerPaddingRate', 0),
                
                # 其他
                'hasTransportSequenceNumber': packet_data.get('hastransportSequenceNumber', False),
            }
            
            # 注意：sendTimestamp 和 arrivalTimeMs 可能不在同一时间基准
            # sendTimestamp 通常是相对时间（从会话开始），arrivalTimeMs 是绝对时间
            # 因此不能直接计算 RTT，需要通过数据包之间的时间差来估算延迟变化
            parsed['rtt_ms'] = 0  # 不直接计算，后续通过数据包间延迟差来估算
            
            return parsed
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return None
    
    def _align_and_extract_features(self) -> List[Dict]:
        """
        按时间对齐 BWE estimations 和数据包，提取特征
        
        策略：
        1. 对于每个 BWE estimation，找到时间窗口内的所有数据包
        2. 从这些数据包中提取特征
        3. 生成输入-输出对
        
        注意：BWE 时间戳和数据包时间戳可能使用不同的基准，需要找到它们的关系
        """
        records = []
        
        if not self.bwe_estimations or not self.packet_infos:
            return records
        
        # 按时间排序
        self.bwe_estimations.sort(key=lambda x: x[0])
        self.packet_infos.sort(key=lambda x: x['arrivalTimeMs'])
        
        # 找到时间基准：使用第一个 BWE 和第一个数据包的时间差
        first_bwe_time = self.bwe_estimations[0][0]
        first_packet_time = self.packet_infos[0]['arrivalTimeMs']
        time_offset = first_packet_time - first_bwe_time
        
        # 为每个 BWE estimation 找到对应的数据包窗口
        packet_idx = 0
        for bwe_time, bwe_value in self.bwe_estimations:
            # 将 BWE 时间转换为数据包时间基准
            bwe_time_abs = bwe_time + time_offset
            
            # 找到时间窗口内的数据包
            # 策略：找到 BWE 时间之前最近的 window_size 个数据包
            window_packets = []
            
            # 从后往前找，找到 BWE 时间之前的数据包
            for i in range(len(self.packet_infos) - 1, -1, -1):
                pkt = self.packet_infos[i]
                if pkt['arrivalTimeMs'] <= bwe_time_abs:
                    window_packets.insert(0, pkt)
                    if len(window_packets) >= self.window_size:
                        break
            
            # 如果找到足够的数据包，提取特征
            if len(window_packets) >= 2:  # 至少需要2个包才能计算一些统计量
                features = self._extract_features_from_packets(window_packets, bwe_time)
                record = {
                    'timestamp': bwe_time,
                    'input_features': features,
                    'output_bandwidth': int(bwe_value),
                    'algorithm': self._extract_algorithm_name()
                }
                records.append(record)
        
        return records
    
    def _extract_features_from_packets(self, packets: List[Dict], bwe_time: int) -> Dict:
        """
        从数据包列表中提取特征
        
        Args:
            packets: 数据包列表
            bwe_time: BWE estimation 的时间戳
        
        Returns:
            特征字典
        """
        if not packets:
            return self._get_default_features()
        
        # 过滤视频包（payloadType 125）
        video_packets = [p for p in packets if p.get('payloadType') == 125]
        
        # 1. 数据包统计
        total_packets = len(packets)
        video_packets_count = len(video_packets)
        total_bytes = sum(p.get('totalSize', 0) for p in packets)
        video_bytes = sum(p.get('totalSize', 0) for p in video_packets)
        
        # 2. RTT 统计（通过数据包间到达时间差估算）
        # 由于 sendTimestamp 和 arrivalTimeMs 时间基准不同，无法直接计算 RTT
        # 这里使用数据包到达间隔作为延迟的近似
        if len(video_packets) >= 2:
            arrival_times = [p.get('arrivalTimeMs', 0) for p in video_packets]
            arrival_times = [t for t in arrival_times if t > 0]
            if len(arrival_times) >= 2:
                inter_arrival_times = [arrival_times[i] - arrival_times[i-1] 
                                      for i in range(1, len(arrival_times))]
                min_rtt = min(inter_arrival_times) if inter_arrival_times else 0
                avg_rtt = sum(inter_arrival_times) / len(inter_arrival_times) if inter_arrival_times else 0
                max_rtt = max(inter_arrival_times) if inter_arrival_times else 0
            else:
                min_rtt = avg_rtt = max_rtt = 0
        else:
            min_rtt = avg_rtt = max_rtt = 0
        
        # 3. 丢包率
        loss_rates = [p.get('lossRates', 0.0) for p in packets]
        avg_loss_rate = sum(loss_rates) / len(loss_rates) if loss_rates else 0.0
        
        # 4. 时间窗口和吞吐量
        if len(packets) >= 2:
            first_pkt = packets[0]
            last_pkt = packets[-1]
            time_window = last_pkt.get('arrivalTimeMs', 0) - first_pkt.get('arrivalTimeMs', 0)
            
            if time_window > 0:
                throughput = (total_bytes * 8 * 1000) / time_window  # bps
            else:
                throughput = 0
        else:
            time_window = 0
            throughput = 0
        
        # 5. 延迟梯度（到达间隔变化率）
        delay_gradient = 0.0
        if len(video_packets) >= 4 and time_window > 0:
            arrival_times = [p.get('arrivalTimeMs', 0) for p in video_packets if p.get('arrivalTimeMs', 0) > 0]
            if len(arrival_times) >= 4:
                inter_arrival_times = [arrival_times[i] - arrival_times[i-1] 
                                      for i in range(1, len(arrival_times))]
                mid = len(inter_arrival_times) // 2
                recent_avg = sum(inter_arrival_times[mid:]) / len(inter_arrival_times[mid:]) if inter_arrival_times[mid:] else 0
                prev_avg = sum(inter_arrival_times[:mid]) / len(inter_arrival_times[:mid]) if inter_arrival_times[:mid] else 0
                if time_window > 0:
                    delay_gradient = (recent_avg - prev_avg) / (time_window / 1000.0)  # ms/s
        
        # 6. 序列号分析（检测丢包）
        seq_nums = [p.get('sequenceNumber', 0) for p in video_packets if p.get('sequenceNumber', 0) > 0]
        if len(seq_nums) >= 2:
            min_seq = min(seq_nums)
            max_seq = max(seq_nums)
            expected_packets = max_seq - min_seq + 1
            if expected_packets > 0:
                packet_loss_rate = 1.0 - (len(seq_nums) / expected_packets)
                packet_loss_rate = max(0.0, min(1.0, packet_loss_rate))
            else:
                packet_loss_rate = 0.0
        else:
            packet_loss_rate = avg_loss_rate
        
        # 7. Pacer 速率（使用最新的）
        latest_pacer_rate = packets[-1].get('pacerPacingRate', 0) if packets else 0
        
        return {
            # 数据包统计
            'total_packets': total_packets,
            'video_packets': video_packets_count,
            'total_bytes': total_bytes,
            'video_bytes': video_bytes,
            
            # RTT 统计
            'min_rtt_ms': min_rtt,
            'avg_rtt_ms': avg_rtt,
            'max_rtt_ms': max_rtt,
            
            # 丢包率
            'loss_rate': avg_loss_rate,
            'packet_loss_rate': packet_loss_rate,
            
            # 时间窗口和吞吐量
            'time_window_ms': time_window,
            'throughput_bps': throughput,
            
            # 延迟梯度
            'delay_gradient_ms_per_s': delay_gradient,
            
            # 当前带宽（pacer rate）
            'current_bandwidth_bps': latest_pacer_rate,
        }
    
    def _get_default_features(self) -> Dict:
        """返回默认特征（当没有数据包时）"""
        return {
            'total_packets': 0,
            'video_packets': 0,
            'total_bytes': 0,
            'video_bytes': 0,
            'min_rtt_ms': 0,
            'avg_rtt_ms': 0,
            'max_rtt_ms': 0,
            'loss_rate': 0.0,
            'packet_loss_rate': 0.0,
            'time_window_ms': 0,
            'throughput_bps': 0,
            'delay_gradient_ms_per_s': 0.0,
            'current_bandwidth_bps': 0,
        }
    
    def _extract_algorithm_name(self) -> str:
        """从文件路径提取算法名称"""
        filename = os.path.basename(self.log_path)
        # 格式：webrtc_<algorithm>.log
        match = re.search(r'webrtc_(\w+)\.log', filename)
        return match.group(1) if match else "Unknown"


def save_parsed_data(records: List[Dict], output_path: str):
    """保存解析的数据到 JSON 文件"""
    if not records:
        print("警告：没有数据可保存")
        return
    
    output_data = {
        "algorithm": records[0].get('algorithm', 'Unknown'),
        "total_records": len(records),
        "data": records
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 已保存 {len(records)} 条记录到 {output_path}")


def print_extractable_fields():
    """打印可提取的输入输出字段说明"""
    print("=" * 80)
    print("可提取的输入输出字段说明")
    print("=" * 80)
    
    print("\n【输出字段】（BWE Estimation）:")
    print("  1. output_bandwidth (int): 带宽估计值（bps）")
    print("  2. timestamp (int): BWE estimation 的时间戳（ms）")
    
    print("\n【输入字段】（从数据包信息提取的特征）:")
    print("\n  数据包统计:")
    print("    - total_packets: 总数据包数量")
    print("    - video_packets: 视频数据包数量（payloadType=125）")
    print("    - total_bytes: 总字节数")
    print("    - video_bytes: 视频数据包字节数")
    
    print("\n  RTT 统计:")
    print("    - min_rtt_ms: 最小往返时延（ms）")
    print("    - avg_rtt_ms: 平均往返时延（ms）")
    print("    - max_rtt_ms: 最大往返时延（ms）")
    
    print("\n  丢包率:")
    print("    - loss_rate: 平均丢包率（从 lossRates 字段）")
    print("    - packet_loss_rate: 基于序列号计算的丢包率")
    
    print("\n  吞吐量和时间窗口:")
    print("    - time_window_ms: 时间窗口大小（ms）")
    print("    - throughput_bps: 吞吐量（bps）")
    
    print("\n  延迟梯度:")
    print("    - delay_gradient_ms_per_s: 延迟梯度（ms/s），表示 RTT 变化率")
    
    print("\n  当前带宽:")
    print("    - current_bandwidth_bps: 当前 pacer pacing rate（bps）")
    
    print("\n【原始数据包字段】（在解析过程中使用，但最终特征中已聚合）:")
    print("  - arrivalTimeMs: 数据包到达时间（ms）")
    print("  - sendTimestamp: 数据包发送时间戳")
    print("  - sequenceNumber: 序列号")
    print("  - ssrc: 同步源标识符")
    print("  - payloadType: 负载类型（111=音频，125=视频）")
    print("  - headerLength: 头部长度（字节）")
    print("  - paddingLength: 填充长度（字节）")
    print("  - payloadSize: 负载大小（字节）")
    print("  - lossRates: 丢包率")
    print("  - pacerPacingRate: Pacer 发送速率（bps）")
    print("  - pacerPaddingRate: Pacer 填充速率（bps）")
    print("  - hasTransportSequenceNumber: 是否有传输序列号")
    
    print("\n" + "=" * 80)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python parse_webrtc_log.py <log_file> [output_file]")
        print("\n示例:")
        print("  python parse_webrtc_log.py share/output/trace/webrtc_BBR.log")
        print("  python parse_webrtc_log.py share/output/trace/webrtc_BBR.log output.json")
        print("\n显示可提取字段:")
        print("  python parse_webrtc_log.py --help-fields")
        sys.exit(1)
    
    if sys.argv[1] == '--help-fields':
        print_extractable_fields()
        sys.exit(0)
    
    log_path = sys.argv[1]
    
    if not os.path.exists(log_path):
        print(f"错误：文件不存在: {log_path}")
        sys.exit(1)
    
    # 确定输出路径
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # 默认输出路径
        log_dir = os.path.dirname(log_path)
        log_name = os.path.basename(log_path)
        algorithm = log_name.replace('webrtc_', '').replace('.log', '')
        output_dir = "share/output/imitation_data"
        output_path = os.path.join(output_dir, f"{algorithm}_from_log.json")
    
    # 解析
    parser = WebRTCLogParser(log_path, window_size=50)
    records = parser.parse()
    
    # 保存
    if records:
        save_parsed_data(records, output_path)
        
        # 打印前几条记录示例
        print(f"\n前 3 条记录示例:")
        for i, record in enumerate(records[:3]):
            print(f"\n记录 {i+1}:")
            print(f"  时间戳: {record['timestamp']} ms")
            print(f"  输出带宽: {record['output_bandwidth']} bps ({record['output_bandwidth']/1000:.2f} kbps)")
            print(f"  输入特征:")
            for key, value in record['input_features'].items():
                print(f"    {key}: {value}")
    else:
        print("警告：未能解析出任何记录")


if __name__ == "__main__":
    main()

