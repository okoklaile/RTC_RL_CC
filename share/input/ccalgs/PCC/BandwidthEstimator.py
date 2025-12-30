import collections
import json
import os
import math

# PCC Vivace Constants
kMinDurationMicro = 2 * 1000 * 1000  # 最小 MI 持续时间 (us) - 实际上适配 PyRTC 可用 200ms
kInitialRate = 300000        # 初始带宽 300kbps
kMinRate = 150000            # 最小带宽
kAlpha = 0.9                 # 吞吐量效用指数
kLossCoefficient = 11.35     # 丢包惩罚系数
kLatencyCoefficient = 900    # 延迟梯度惩罚系数
kGradientStepSize = 0.05     # 每次调整的步长 (5%)
kMonitorIntervalMs = 200     # MI 时长 (ms)

class Estimator(object):
    def __init__(self, enable_data_collection: bool = False, output_dir: str = "share/output/imitation_data"):
        """
        初始化 PCC 估计器
        
        Args:
            enable_data_collection: 是否启用数据收集（用于模仿学习）
            output_dir: 数据输出目录
        """
        self.enable_data_collection = enable_data_collection
        self.output_dir = output_dir
        self.data_records = []
        self.call_count = 0
        
        if self.enable_data_collection:
            os.makedirs(output_dir, exist_ok=True)
        self.start_time_ms = -1
        self.current_rate = kInitialRate
        self.last_rate = kInitialRate
        
        # Vivace State
        self.utility_history = []  # 存储 (rate, utility)
        self.gradient = 1          # 初始梯度方向 (1: 增加, -1: 减少)
        self.step_size = kGradientStepSize
        self.mi_start_ms = -1
        
        # RTT Tracking for gradient
        self.prev_avg_rtt = 0
        self.curr_avg_rtt = 0

    def report_states(self, stats: dict):
        '''
        收集数据包统计信息
        '''
        pkt = stats
        packet_info = PacketInfo()
        packet_info.payload_type = pkt["payload_type"]
        packet_info.ssrc = pkt["ssrc"]
        packet_info.sequence_number = pkt["sequence_number"]
        packet_info.send_timestamp = pkt["send_time_ms"]
        packet_info.receive_timestamp = pkt["arrival_time_ms"]
        packet_info.payload_size = pkt["payload_size"]
        packet_info.size = pkt["header_length"] + pkt["payload_size"] + pkt["padding_length"]
        
        now_ms = packet_info.receive_timestamp
        if self.start_time_ms == -1:
            self.start_time_ms = now_ms
            self.mi_start_ms = now_ms

        self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self) -> int:
        '''
        PCC 核心逻辑：在 Monitor Interval 结束时计算 Utility 并调整速率
        '''
        if not self.packets_list:
            return int(self.current_rate)

        now_ms = self.packets_list[-1].receive_timestamp
        duration = now_ms - self.mi_start_ms

        # 检查 Monitor Interval 是否结束 (例如 200ms 或 1个 RTT)
        # 这里为了简化适配，使用固定 200ms 作为决策周期
        if duration < kMonitorIntervalMs:
            return int(self.current_rate)

        # 1. 计算当前 MI 的统计数据
        throughput, loss_rate, avg_rtt = self.compute_stats(duration)
        
        # 2. 计算效用 (Utility)
        # 延迟梯度 (Latency Gradient): (CurrRTT - PrevRTT) / duration
        # 为了数值稳定性，单位统一处理
        rtt_gradient = 0
        if self.prev_avg_rtt > 0:
            rtt_gradient = (avg_rtt - self.prev_avg_rtt) / (duration / 1000.0) # ms/s
        
        # 简单的噪声过滤
        if abs(rtt_gradient) < 5: 
            rtt_gradient = 0

        # Vivace Utility Function
        # U = T^0.9 - 11.35 * T * L - 900 * T * rtt_gradient
        # T 单位: Mbps (避免数值过大)
        t_mbps = throughput / 1000000.0
        
        utility = (math.pow(t_mbps, kAlpha) 
                   - kLossCoefficient * t_mbps * loss_rate 
                   - kLatencyCoefficient * t_mbps * max(0, rtt_gradient))

        # 3. 速率调整 (Gradient Ascent 简化版)
        self.update_rate(utility)

        # 4. 更新状态以准备下一个 MI
        self.prev_avg_rtt = avg_rtt
        self.mi_start_ms = now_ms
        self.packets_list = [] # 清空当前 MI 数据
        
        return int(self.current_rate)

    def compute_stats(self, duration_ms):
        '''
        计算 MI 内的吞吐量、丢包率、平均 RTT
        '''
        total_bytes = 0
        rtt_sum = 0
        count = 0
        min_seq = float('inf')
        max_seq = float('-inf')
        
        valid_pkts = 0

        for pkt in self.packets_list:
            total_bytes += pkt.size
            rtt = pkt.receive_timestamp - pkt.send_timestamp
            rtt_sum += rtt
            count += 1
            
            if pkt.payload_type == 126: # 假设 126 是视频流
                valid_pkts += 1
                if pkt.sequence_number < min_seq: min_seq = pkt.sequence_number
                if pkt.sequence_number > max_seq: max_seq = pkt.sequence_number

        # Throughput (bps)
        throughput = (total_bytes * 8) / (duration_ms / 1000.0)
        
        # Avg RTT (ms)
        avg_rtt = rtt_sum / count if count > 0 else 0
        
        # Loss Rate
        loss_rate = 0
        if valid_pkts > 0 and (max_seq - min_seq) > 0:
            expected = max_seq - min_seq + 1
            loss_rate = 1.0 - (valid_pkts / expected)
            loss_rate = max(0.0, min(1.0, loss_rate))

        return throughput, loss_rate, avg_rtt

    def update_rate(self, current_utility):
        '''
        基于 Utility 变化调整速率
        '''
        if len(self.utility_history) > 0:
            prev_rate, prev_utility = self.utility_history[-1]
            
            # 如果 Utility 增加，继续沿当前梯度方向走
            if current_utility > prev_utility:
                # 保持方向，步长可能加速 (这里保持固定步长简化)
                pass 
            else:
                # Utility 减少，反转方向
                self.gradient *= -1
                # 可选：减小步长以收敛
                
        # 记录历史
        self.utility_history.append((self.current_rate, current_utility))
        if len(self.utility_history) > 10:
            self.utility_history.pop(0)

        # 应用调整
        # NewRate = OldRate * (1 + sign * step)
        change = self.gradient * self.step_size
        self.current_rate = self.current_rate * (1 + change)
        
        # 边界限制
        self.current_rate = max(self.current_rate, kMinRate)
        # PyRTC 模拟中通常不需要设上限，或设一个物理上限
        self.current_rate = min(self.current_rate, 100 * 1000 * 1000) # 100 Mbps max




    def _extract_features_from_packets(self, packets_list) -> dict:
        """
        从数据包列表中提取通用特征（适用于所有算法）
        
        Args:
            packets_list: 数据包列表（PacketInfo 对象列表）
        
        Returns:
            特征字典
        """
        if not packets_list:
            return self._get_default_features()
        
        # 过滤视频包
        video_packets = [p for p in packets_list if hasattr(p, 'payload_type') and p.payload_type == 125]
        
        # 1. 数据包统计
        total_packets = len(packets_list)
        video_packets_count = len(video_packets)
        total_bytes = sum(p.size for p in packets_list if hasattr(p, 'size'))
        video_bytes = sum(p.size for p in video_packets if hasattr(p, 'size'))
        
        # 2. RTT 统计
        rtt_list = []
        for pkt in video_packets:
            if hasattr(pkt, 'receive_timestamp') and hasattr(pkt, 'send_timestamp'):
                if pkt.receive_timestamp > 0 and pkt.send_timestamp > 0:
                    rtt = pkt.receive_timestamp - pkt.send_timestamp
                    if rtt > 0:
                        rtt_list.append(rtt)
        
        min_rtt = min(rtt_list) if rtt_list else 0
        avg_rtt = sum(rtt_list) / len(rtt_list) if rtt_list else 0
        max_rtt = max(rtt_list) if rtt_list else 0
        
        # 如果没有 RTT 数据，使用数据包到达间隔作为近似
        if min_rtt == 0 and len(video_packets) >= 2:
            arrival_times = [p.receive_timestamp for p in video_packets if hasattr(p, 'receive_timestamp')]
            arrival_times = [t for t in arrival_times if t > 0]
            if len(arrival_times) >= 2:
                inter_arrival_times = [arrival_times[i] - arrival_times[i-1] 
                                      for i in range(1, len(arrival_times))]
                if inter_arrival_times:
                    min_rtt = min(inter_arrival_times)
                    avg_rtt = sum(inter_arrival_times) / len(inter_arrival_times)
                    max_rtt = max(inter_arrival_times)
        
        # 3. 时间窗口和吞吐量
        if len(packets_list) >= 2:
            first_pkt = packets_list[0]
            last_pkt = packets_list[-1]
            if hasattr(first_pkt, 'receive_timestamp') and hasattr(last_pkt, 'receive_timestamp'):
                time_window = last_pkt.receive_timestamp - first_pkt.receive_timestamp
                if time_window > 0:
                    throughput = (total_bytes * 8 * 1000) / time_window  # bps
                else:
                    throughput = 0
            else:
                time_window = 0
                throughput = 0
        else:
            time_window = 0
            throughput = 0
        
        # 4. 延迟梯度（到达间隔变化率）
        delay_gradient = 0.0
        if len(video_packets) >= 4 and time_window > 0:
            arrival_times = [p.receive_timestamp for p in video_packets if hasattr(p, 'receive_timestamp')]
            arrival_times = [t for t in arrival_times if t > 0]
            if len(arrival_times) >= 4:
                inter_arrival_times = [arrival_times[i] - arrival_times[i-1] 
                                      for i in range(1, len(arrival_times))]
                mid = len(inter_arrival_times) // 2
                recent_avg = sum(inter_arrival_times[mid:]) / len(inter_arrival_times[mid:]) if inter_arrival_times[mid:] else 0
                prev_avg = sum(inter_arrival_times[:mid]) / len(inter_arrival_times[:mid]) if inter_arrival_times[:mid] else 0
                if time_window > 0:
                    delay_gradient = (recent_avg - prev_avg) / (time_window / 1000.0)  # ms/s
        
        # 5. 序列号分析（检测丢包）
        seq_nums = [p.sequence_number for p in video_packets if hasattr(p, 'sequence_number') and p.sequence_number is not None]
        packet_loss_rate = 0.0
        if len(seq_nums) >= 2:
            min_seq = min(seq_nums)
            max_seq = max(seq_nums)
            expected_packets = max_seq - min_seq + 1
            if expected_packets > 0:
                packet_loss_rate = 1.0 - (len(seq_nums) / expected_packets)
                packet_loss_rate = max(0.0, min(1.0, packet_loss_rate))
        
        # 6. 返回通用特征（适用于所有算法）
        return {
            'total_packets': total_packets,
            'video_packets': video_packets_count,
            'total_bytes': total_bytes,
            'video_bytes': video_bytes,
            'min_rtt_ms': min_rtt,
            'avg_rtt_ms': avg_rtt,
            'max_rtt_ms': max_rtt,
            'loss_rate': packet_loss_rate,
            'time_window_ms': time_window,
            'throughput_bps': throughput,
            'delay_gradient_ms_per_s': delay_gradient,
            'current_bandwidth_bps': current_rate,
        }
    
    def _get_default_features(self) -> dict:
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
            'time_window_ms': 0,
            'throughput_bps': 0,
            'delay_gradient_ms_per_s': 0.0,
            'current_bandwidth_bps': current_rate,
        }
    
    def _save_data_record(self, features: dict, bandwidth: int):
        """保存一条数据记录"""
        record = {
            "timestamp": self.call_count,
            "input_features": features,
            "output_bandwidth": bandwidth,
            "algorithm": "PCC"
        }
        self.data_records.append(record)
        self.call_count += 1
    
    def save_data(self, filename: str = None, auto_increment: bool = True):
        """保存收集的数据到 JSON 文件"""
        if not self.enable_data_collection or not self.data_records:
            return None
        
        if filename is None:
            filename = "PCC_imitation_data.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if auto_increment and os.path.exists(filepath):
            base_name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(filepath):
                filename = f"{base_name}_{counter}{ext}"
                filepath = os.path.join(self.output_dir, filename)
                counter += 1
        
        output_data = {
            "algorithm": "PCC",
            "total_records": len(self.data_records),
            "data": self.data_records
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"已保存 {len(self.data_records)} 条数据到 {filepath}")
        return filepath
    
    def __del__(self):
        """析构函数：自动保存数据"""
        if hasattr(self, 'enable_data_collection') and self.enable_data_collection:
            if hasattr(self, 'data_records') and len(self.data_records) > 0:
                try:
                    self.save_data(auto_increment=True)
                except Exception:
                    pass
class PacketInfo:
    def __init__(self, enable_data_collection: bool = False, output_dir: str = "share/output/imitation_data"):
        self.payload_type = None
        self.sequence_number = None
        self.send_timestamp = None
        self.ssrc = None
        self.padding_length = None
        self.header_length = None
        self.receive_timestamp = None
        self.payload_size = None
        self.size = None