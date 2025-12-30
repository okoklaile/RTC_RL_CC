import collections
import json
import os

# COPA Constants
kDefaultDelta = 0.5          # 默认 delta (0.5 为论文推荐值)
kMinBitrate = 300000         # 300 kbps
kInitBitrate = 1000000       # 1 Mbps
kMaxBitrate = 50 * 1000000   # 50 Mbps

class Estimator(object):
    def __init__(self, enable_data_collection: bool = False, output_dir: str = "share/output/imitation_data"):
        """
        初始化 Copa 估计器
        
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
        self.start_time = -1
        
        # COPA State variables
        self.current_bitrate = kInitBitrate
        self.delta = kDefaultDelta
        self.velocity = 1.0       # 速率调整速度因子 (v)
        self.direction = 0        # 0: None, 1: Increase, -1: Decrease
        self.same_direction_count = 0
        
        # RTT and Delay tracking
        self.min_rtt = float('inf')
        self.avg_rtt = 0
        self.queuing_delay = 0
        self.now_ms = 0

    def reset(self):
        self.packets_list = []
        self.current_bitrate = kInitBitrate
        self.delta = kDefaultDelta
        self.velocity = 1.0
        self.direction = 0
        self.same_direction_count = 0
        self.min_rtt = float('inf')
        self.start_time = -1

    def report_states(self, stats: dict):
        pkt = stats
        packet_info = PacketInfo()
        packet_info.payload_type = pkt["payload_type"]
        packet_info.ssrc = pkt["ssrc"]
        packet_info.sequence_number = pkt["sequence_number"]
        packet_info.send_timestamp = pkt["send_time_ms"]
        packet_info.receive_timestamp = pkt["arrival_time_ms"]
        packet_info.payload_size = pkt["payload_size"]
        packet_info.size = pkt["header_length"] + pkt["payload_size"] + pkt["padding_length"]
        
        self.now_ms = packet_info.receive_timestamp
        if self.start_time == -1:
            self.start_time = self.now_ms

        self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self) -> int:
        if not self.packets_list:
            return int(self.current_bitrate)

        # 1. 更新 RTT 和排队延迟
        self.update_rtt_stats()

        # 2. 计算目标速率 (Target Rate)
        # 公式: lambda_target = 1 / (delta * d_queue)
        # 为了单位适配 (bits/s), 我们假设一个平均包大小或直接使用比例系数
        # Target_Rate (bps) = (MTU_bits) / (delta * queuing_delay_sec)
        mtu_bits = 1200 * 8 # 假设平均包大小 1200 字节
        
        if self.queuing_delay <= 0.002: # 延迟极低 (2ms以下)
            # 几乎没有排队，允许倍增或最大化利用
            target_bitrate = self.current_bitrate * 2
        else:
            target_bitrate = mtu_bits / (self.delta * self.queuing_delay)

        # 3. 更新速率 (基于 Target Rate)
        self.update_rate(target_bitrate)

        # 4. 更新 Delta (竞争模式)
        self.update_delta()

        bandwidth = int(self.current_bitrate)
        
        # 数据收集：在清空列表之前记录实际使用的数据包和特征
        if self.enable_data_collection and self.packets_list:
            packets_copy = list(self.packets_list)
            features = self._extract_features_from_packets(packets_copy)
            self._save_data_record(features, bandwidth)
        elif self.enable_data_collection:
            features = self._get_default_features()
            self._save_data_record(features, bandwidth)
        
        # 清空数据包缓存
        self.packets_list = []
        
        return bandwidth

    def update_rtt_stats(self):
        curr_rtt_sum = 0
        count = 0
        for pkt in self.packets_list:
            # 修改点：增加 payload_type 检查，仅使用视频包计算 RTT
            if pkt.payload_type == 125: 
                rtt = pkt.receive_timestamp - pkt.send_timestamp
                if rtt > 0:
                    curr_rtt_sum += rtt
                    count += 1
                    # 维护最小 RTT (min_rtt)
                    if rtt < self.min_rtt:
                        self.min_rtt = rtt
        
        if count > 0:
            self.avg_rtt = curr_rtt_sum / count
            self.queuing_delay = (self.avg_rtt - self.min_rtt) / 1000.0
            
            # 简单的过期机制: 如果很久没更新 min_rtt，可能网络路径变了
            # 标准 Copa 不包含此逻辑，但在长期运行中推荐加入
            # 这里保持纯粹，不主动重置 min_rtt，依靠 Copa+ 来处理 Bufferbloat

    def update_rate(self, target_bitrate):
        # 类似 TCP 的 AI (Additive Increase) 和 MD (Multiplicative Decrease)
        # 但 Copa 使用速度因子 v 来加速收敛
        
        rate_change_step = (kInitBitrate * 0.05) * self.velocity
        
        if self.current_bitrate < target_bitrate:
            # 需要增加速率
            new_direction = 1
            self.current_bitrate += rate_change_step
        else:
            # 需要减少速率
            new_direction = -1
            self.current_bitrate = max(kMinBitrate, self.current_bitrate - rate_change_step)

        # 更新速度因子 v
        if new_direction == self.direction:
            self.same_direction_count += 1
            if self.same_direction_count > 3:
                # 连续同向调整 3 次，加速 (v *= 2 or v += 1)
                self.velocity = min(self.velocity * 2, 8.0)
                self.same_direction_count = 0 # 重置计数或保留取决于策略，这里重置以平滑
        else:
            # 方向反转，重置速度
            self.velocity = 1.0
            self.same_direction_count = 0
            self.direction = new_direction
            
        self.current_bitrate = min(self.current_bitrate, kMaxBitrate)

    def update_delta(self):
        """
        竞争模式: 动态调整 Delta
        如果排队延迟持续很高 (被 TCP 填满)，减小 Delta 以提高 Target Rate。
        如果排队延迟很低，增加 Delta 以降低延迟。
        """
        is_queue_busy = self.queuing_delay > 0.01 # >10ms
        
        if is_queue_busy:
            # 更加激进 (Competitve)
            self.delta = max(0.1, self.delta * 0.98)
        else:
            # 回归保守 (Delay-sensitive)
            self.delta = min(kDefaultDelta, self.delta * 1.05)



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
            'current_bandwidth_bps': current_bitrate,
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
            'current_bandwidth_bps': current_bitrate,
        }
    
    def _save_data_record(self, features: dict, bandwidth: int):
        """保存一条数据记录"""
        record = {
            "timestamp": self.call_count,
            "input_features": features,
            "output_bandwidth": bandwidth,
            "algorithm": "Copa"
        }
        self.data_records.append(record)
        self.call_count += 1
    
    def save_data(self, filename: str = None, auto_increment: bool = True):
        """保存收集的数据到 JSON 文件"""
        if not self.enable_data_collection or not self.data_records:
            return None
        
        if filename is None:
            filename = "Copa_imitation_data.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if auto_increment and os.path.exists(filepath):
            base_name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(filepath):
                filename = f"{base_name}_{counter}{ext}"
                filepath = os.path.join(self.output_dir, filename)
                counter += 1
        
        output_data = {
            "algorithm": "Copa",
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