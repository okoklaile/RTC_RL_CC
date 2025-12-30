import collections
import json
import os
import math

# CUBIC Constants
kCubicC = 0.4           # CUBIC 系数 C (通常为 0.4)
kBeta = 0.7             # 乘性减小因子 (丢包后带宽降为原来的 0.7)
kMinBitrate = 300000    # 最小带宽 (300 kbps)
kInitBitrate = 1000000  # 初始带宽 (1 Mbps)
kTimeInterval = 200     # 统计时间窗口 (ms)

class Estimator(object):
    def __init__(self, enable_data_collection: bool = False, output_dir: str = "share/output/imitation_data"):
        """
        初始化 Cubic 估计器
        
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
        
        self.packets_list = []
        self.now_ms = 0
        
        # CUBIC state variables
        self.last_bandwidth_estimation = kInitBitrate
        self.w_max = kInitBitrate          # W_max: 上次发生拥塞时的饱和带宽
        self.last_congestion_time = -1     # 上次发生拥塞的时间 (ms)
        self.k = 0                         # K: 达到 W_max 所需的时间 (s)
        self.start_time = -1               # 算法开始时间

    def reset(self):
        self.packets_list = []
        self.now_ms = 0
        self.last_bandwidth_estimation = kInitBitrate
        self.w_max = kInitBitrate
        self.last_congestion_time = -1
        self.k = 0
        self.start_time = -1

    def report_states(self, stats: dict):
        '''
        收集接收到的数据包信息
        '''
        pkt = stats
        packet_info = PacketInfo()
        packet_info.payload_type = pkt["payload_type"]
        packet_info.ssrc = pkt["ssrc"]
        packet_info.sequence_number = pkt["sequence_number"]
        packet_info.send_timestamp = pkt["send_time_ms"]
        packet_info.receive_timestamp = pkt["arrival_time_ms"]
        packet_info.padding_length = pkt["padding_length"]
        packet_info.header_length = pkt["header_length"]
        packet_info.payload_size = pkt["payload_size"]
        packet_info.size = pkt["header_length"] + pkt["payload_size"] + pkt["padding_length"]
        
        # 更新当前系统时间
        self.now_ms = packet_info.receive_timestamp
        if self.start_time == -1:
            self.start_time = self.now_ms
            # 注意：last_congestion_time 应保持为 -1，只有在真正发生拥塞事件时才设置

        self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self) -> int:
        '''
        计算估算带宽 (CUBIC Logic)
        '''
        # 1. 计算丢包率作为拥塞信号
        loss_rate = self.calculate_loss_rate()
        
        # 2. 如果检测到丢包 (拥塞事件)
        # 这里设定丢包率阈值为 2% (0.02)
        if loss_rate > 0.02:
            self.handle_congestion_event()
        else:
            self.cubic_update()

        bandwidth = int(self.last_bandwidth_estimation)
        
        # 数据收集：在清空列表之前记录实际使用的数据包和特征
        if self.enable_data_collection and self.packets_list:
            packets_copy = list(self.packets_list)
            features = self._extract_features_from_packets(packets_copy)
            self._save_data_record(features, bandwidth)
        elif self.enable_data_collection:
            features = self._get_default_features()
            self._save_data_record(features, bandwidth)
        
        # 清空过期数据包，准备下一轮统计
        # 注意：在 PyRTC 中通常每次 get_estimated_bandwidth 后会清空 list，
        # 或者保留一部分。这里简单起见，每次计算后清空，模拟周期性更新。
        self.packets_list = []
        
        return bandwidth

    def calculate_loss_rate(self):
        '''
        计算当前数据包列表中的丢包率
        '''
        if len(self.packets_list) == 0:
            return 0
            
        min_seq = self.packets_list[0].sequence_number
        max_seq = self.packets_list[0].sequence_number
        valid_packets_count = 0
        
        # 仅统计视频包 (payload_type 126 通常是 PyRTC 默认的视频流)
        for pkt in self.packets_list:
            if pkt.payload_type == 125:
                valid_packets_count += 1
                if pkt.sequence_number < min_seq:
                    min_seq = pkt.sequence_number
                if pkt.sequence_number > max_seq:
                    max_seq = pkt.sequence_number
                    
        total_expected = max_seq - min_seq + 1
        if total_expected <= 0:
            return 0
            
        # 简单的丢包率计算: 1 - (实际收到 / 理论应收)
        # 注意：这里未处理 seq 回绕的情况，但在短时间窗口内通常没问题
        return 1.0 - (valid_packets_count / total_expected)

    def handle_congestion_event(self):
        '''
        处理拥塞事件：乘性减小，计算 K 值
        '''
        # 记录当前的 W_max
        # 为了避免连续丢包导致带宽降得太低，可以加一个限制或平滑，这里采用标准逻辑
        self.w_max = self.last_bandwidth_estimation
        
        # Multiplicative Decrease: Reduce bandwidth
        self.last_bandwidth_estimation = max(self.last_bandwidth_estimation * kBeta, kMinBitrate)
        
        # 更新拥塞时间
        self.last_congestion_time = self.now_ms
        
        # 计算 K 值
        # 公式: K = (W_max * (1 - beta) / C)^(1/3)
        # 注意: 这里我们将带宽直接作为 Window 处理，单位需要统一。
        # 为了让公式在 Mbps 级别工作得更好，可以将 bps 转为 Mbps 计算，最后再转回 bps，
        # 或者调整 C 的量级。这里将带宽单位转换为 Mbps 进行计算以适配通常的 C=0.4。
        w_max_mbps = self.w_max / 1000000.0
        current_mbps = self.last_bandwidth_estimation / 1000000.0
        
        # K = cuberoot((W_max - Current) / C)
        if w_max_mbps > current_mbps:
            self.k = math.pow((w_max_mbps - current_mbps) / kCubicC, 1.0/3.0)
        else:
            self.k = 0

    def cubic_update(self):
        '''
        CUBIC 带宽增长更新
        W(t) = C(t - K)^3 + W_max
        '''
        if self.last_congestion_time == -1:
            # 刚启动还未发生过拥塞，可以像 TCP Slow Start 一样指数增长
            # 或者简单地线性增长。这里为了简化，使用 CUBIC 曲线，假设 t 从 0 开始
            t_sec = (self.now_ms - self.start_time) / 1000.0
        else:
            # t: 距离上次拥塞的时间 (秒)
            t_sec = (self.now_ms - self.last_congestion_time) / 1000.0
            
        # CUBIC 公式计算 (使用 Mbps)
        w_max_mbps = self.w_max / 1000000.0
        
        # W(t) = C * (t - K)^3 + W_max
        target_mbps = kCubicC * math.pow(t_sec - self.k, 3) + w_max_mbps
        
        # 将 Mbps 转回 bps
        target_bps = target_mbps * 1000000.0
        
        # 确保不低于最小带宽
        self.last_bandwidth_estimation = max(target_bps, kMinBitrate)

        # (可选) 凸区域（Concave region）限制
        # 标准 CUBIC 在 W(t) < W_max 时是凸的，之后是凹的。
        # TCP Friendly 模式：如果 CUBIC 计算出的带宽小于标准 TCP Reno 的增长，通常会使用 Reno。
        # 这里为了保持纯粹的 CUBIC 演示，暂不加入 TCP Friendly 检查。




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
            'current_bandwidth_bps': last_bandwidth_estimation,
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
            'current_bandwidth_bps': last_bandwidth_estimation,
        }
    
    def _save_data_record(self, features: dict, bandwidth: int):
        """保存一条数据记录"""
        record = {
            "timestamp": self.call_count,
            "input_features": features,
            "output_bandwidth": bandwidth,
            "algorithm": "Cubic"
        }
        self.data_records.append(record)
        self.call_count += 1
    
    def save_data(self, filename: str = None, auto_increment: bool = True):
        """保存收集的数据到 JSON 文件"""
        if not self.enable_data_collection or not self.data_records:
            return None
        
        if filename is None:
            filename = "Cubic_imitation_data.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if auto_increment and os.path.exists(filepath):
            base_name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(filepath):
                filename = f"{base_name}_{counter}{ext}"
                filepath = os.path.join(self.output_dir, filename)
                counter += 1
        
        output_data = {
            "algorithm": "Cubic",
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
        self.sequence_number = None  # int
        self.send_timestamp = None   # int, ms
        self.ssrc = None             # int
        self.padding_length = None   # int, B
        self.header_length = None    # int, B
        self.receive_timestamp = None # int, ms
        self.payload_size = None     # int, B
        self.size = None             # int, B