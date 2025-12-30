import collections
import math
import json
import os

# BBR Constants
kProbeRTTInterval = 10000     # 10s
kProbeRTTDuration = 200       # 200ms
kMinRttWindow = 10000         # 10s
kBtlBwWindowMs = 10000        # 10s

# Gains
kHighGain = 2.885             # 2/ln(2)
kDrainGain = 1.0 / kHighGain  # ln(2)/2
kPacingGainCycle = [1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

kMinBitrate = 150000          # 150 kbps
kInitBitrate = 300000         # 300 kbps (降低初始值，给 Startup 留空间)
kMaxBitrate = 30 * 1000000    # 30 Mbps (防止仿真环境被撑爆)

class BBRState:
    STARTUP = 0
    DRAIN = 1
    PROBE_BW = 2
    PROBE_RTT = 3

class Estimator(object):
    def __init__(self, enable_data_collection: bool = False, output_dir: str = "share/output/imitation_data"):
        """
        初始化 BBR 估计器
        
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
        
        self.reset()

    def reset(self):
        self.packets_list = []
        self.start_time = -1
        self.now_ms = 0

        # BBR State
        self.state = BBRState.STARTUP
        self.pacing_gain = kHighGain
        self.current_bitrate = kInitBitrate
        
        # BtlBw (Max Bandwidth)
        self.btl_bw = 0
        self.btl_bw_filter = WindowedMaxFilter(kBtlBwWindowMs)
        
        # RTprop (Min RTT)
        self.rt_prop = float('inf')
        self.rt_prop_stamp = -1 

        # Cycle logic
        self.cycle_idx = 0
        self.cycle_start_time = -1
        
        # Startup logic
        self.full_bw_reached = False
        self.last_startup_bw = 0
        self.rounds_without_growth = 0
        
        # ProbeRTT logic
        self.probe_rtt_start_ms = -1

    def report_states(self, stats: dict):
        pkt = stats
        packet_info = PacketInfo()
        packet_info.payload_type = pkt["payload_type"]
        packet_info.sequence_number = pkt["sequence_number"]
        packet_info.send_timestamp = pkt["send_time_ms"]
        packet_info.receive_timestamp = pkt["arrival_time_ms"]
        packet_info.size = pkt["header_length"] + pkt["payload_size"] + pkt["padding_length"]
        
        self.now_ms = packet_info.receive_timestamp
        if self.start_time == -1:
            self.start_time = self.now_ms
            self.rt_prop_stamp = self.now_ms
            self.cycle_start_time = self.now_ms

        self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self) -> int:
        if not self.packets_list:
            bandwidth = int(self.current_bitrate)
            if self.enable_data_collection:
                self._save_data_record({}, bandwidth)
            return bandwidth

        # 1. 更新测量模型 (仅使用视频包)
        self.update_model_and_stats()

        # 2. 状态机更新
        self.update_control_state()

        # 3. 计算目标速率
        # 如果 BtlBw 未建立，使用当前速率作为基准进行增长
        ref_bw = self.btl_bw if self.btl_bw > 0 else self.current_bitrate
        target_rate = self.pacing_gain * ref_bw

        # 4. 边界限制
        self.current_bitrate = max(kMinBitrate, target_rate)
        self.current_bitrate = min(self.current_bitrate, kMaxBitrate)
        
        # ProbeRTT 期间限制 (Cwnd = 4 MSS, approx 0.5 * BtlBw)
        if self.state == BBRState.PROBE_RTT and self.btl_bw > 0:
            limit = max(kMinBitrate, self.btl_bw * 0.5)
            self.current_bitrate = min(self.current_bitrate, limit)

        bandwidth = int(self.current_bitrate)
        
        # 数据收集：在清空列表之前记录实际使用的数据包和特征
        if self.enable_data_collection:
            # 保存当前 packets_list 的副本（因为即将被清空）
            packets_copy = list(self.packets_list)
            features = self._extract_features_from_packets(packets_copy)
            self._save_data_record(features, bandwidth)

        # 清空列表，准备下一轮
        self.packets_list = []
        return bandwidth

    def update_model_and_stats(self):
        # 过滤视频包 (Payload 126)
        video_packets = [p for p in self.packets_list if p.payload_type == 125]
        if not video_packets:
            return

        # 更新 BtlBw
        if len(video_packets) >= 2:
            total_bytes = sum([p.size for p in video_packets]) * 8 
            duration = video_packets[-1].receive_timestamp - video_packets[0].receive_timestamp
            
            # 保护: 防止 duration 为 0 (同一 tick 到达)
            if duration <= 0:
                duration = 1 # 设为 1ms 避免除零

            sample_rate = total_bytes / (duration / 1000.0)
            self.btl_bw_filter.update(sample_rate, self.now_ms)
        
        self.btl_bw = self.btl_bw_filter.get_best()

        # 更新 RTprop
        for pkt in video_packets:
            rtt = pkt.receive_timestamp - pkt.send_timestamp
            if rtt > 0:
                if rtt <= self.rt_prop:
                    self.rt_prop = rtt
                    self.rt_prop_stamp = self.now_ms
                else:
                    # 窗口过期逻辑 (10s)
                    if self.now_ms - self.rt_prop_stamp > kMinRttWindow:
                        # 不在此处强制重置，依赖 ProbeRTT
                        pass
    
    def update_control_state(self):
        # 1. Check ProbeRTT Entry
        # 只有在有有效 RTprop 时才检查过期
        if self.rt_prop != float('inf'):
            if (self.state != BBRState.PROBE_RTT and 
                self.now_ms - self.rt_prop_stamp > kProbeRTTInterval):
                self.enter_probe_rtt()
                return

        # 2. Check ProbeRTT Exit
        if self.state == BBRState.PROBE_RTT:
            if self.now_ms - self.probe_rtt_start_ms > kProbeRTTDuration:
                self.exit_probe_rtt()
            return 

        # 3. State Transitions
        if self.state == BBRState.STARTUP:
            if self.check_full_bandwidth_reached():
                self.state = BBRState.DRAIN
                self.pacing_gain = kDrainGain
        
        elif self.state == BBRState.DRAIN:
            # 简化版 Drain 退出：当 pacing rate 降至 BtlBw 以下
            if self.current_bitrate <= self.btl_bw:
                self.state = BBRState.PROBE_BW
                self.pacing_gain = 1.0
                self.cycle_idx = 0
                self.cycle_start_time = self.now_ms

        elif self.state == BBRState.PROBE_BW:
            self.update_probe_bw_cycle()

    def enter_probe_rtt(self):
        self.state = BBRState.PROBE_RTT
        self.pacing_gain = 1.0
        self.probe_rtt_start_ms = self.now_ms

    def exit_probe_rtt(self):
        self.rt_prop_stamp = self.now_ms
        if self.full_bw_reached:
            self.state = BBRState.PROBE_BW
            self.pacing_gain = 1.0
            self.cycle_start_time = self.now_ms # 重置 cycle 计时
        else:
            self.state = BBRState.STARTUP
            self.pacing_gain = kHighGain

    def check_full_bandwidth_reached(self):
        # 如果 BtlBw 还没测出来，继续 Startup
        if self.btl_bw == 0:
            return False

        # 辅助退出条件：如果当前速率已经很大 (接近上限) 或者检测到明显丢包
        # 这里仅使用带宽平稳判定
        if self.last_startup_bw == 0:
            self.last_startup_bw = self.btl_bw
            return False
        
        # 增长阈值 25%
        if self.btl_bw >= self.last_startup_bw * 1.25:
            self.last_startup_bw = self.btl_bw
            self.rounds_without_growth = 0
            return False
        else:
            self.rounds_without_growth += 1
            # 连续 3 次检查 (每次 get_bwe 调用算一次检查，稍微有点频繁，但能防爆)
            if self.rounds_without_growth >= 3:
                self.full_bw_reached = True
                return True
        return False

    def update_probe_bw_cycle(self):
        # 保护: 如果 rt_prop 无效，默认 200ms
        rtt = self.rt_prop if self.rt_prop != float('inf') else 200
        phase_duration = max(rtt, 200)
        
        if self.now_ms - self.cycle_start_time > phase_duration:
            self.cycle_idx = (self.cycle_idx + 1) % len(kPacingGainCycle)
            self.cycle_start_time = self.now_ms
            self.pacing_gain = kPacingGainCycle[self.cycle_idx]
    
    def _extract_features_from_packets(self, packets_list) -> dict:
        """
        从数据包列表中提取特征
        这些特征与算法内部使用的特征一致
        
        Args:
            packets_list: 数据包列表（PacketInfo 对象列表）
        
        Returns:
            特征字典
        """
        if not packets_list:
            return self._get_default_features()
        
        # 过滤视频包
        video_packets = [p for p in packets_list if p.payload_type == 125]
        
        # 1. 数据包统计
        total_packets = len(packets_list)
        video_packets_count = len(video_packets)
        total_bytes = sum(p.size for p in packets_list)
        video_bytes = sum(p.size for p in video_packets)
        
        # 2. RTT 统计
        # 计算数据包 RTT（receive_timestamp - send_timestamp）
        rtt_list = []
        for pkt in video_packets:
            if pkt.receive_timestamp > 0 and pkt.send_timestamp > 0:
                rtt = pkt.receive_timestamp - pkt.send_timestamp
                if rtt > 0:
                    rtt_list.append(rtt)
        
        min_rtt = min(rtt_list) if rtt_list else 0
        avg_rtt = sum(rtt_list) / len(rtt_list) if rtt_list else 0
        max_rtt = max(rtt_list) if rtt_list else 0
        
        # 如果没有 RTT 数据，使用数据包到达间隔作为近似
        if min_rtt == 0 and len(video_packets) >= 2:
            arrival_times = [p.receive_timestamp for p in video_packets]
            inter_arrival_times = [arrival_times[i] - arrival_times[i-1] 
                                  for i in range(1, len(arrival_times))]
            if inter_arrival_times:
                min_rtt = min(inter_arrival_times)
                avg_rtt = sum(inter_arrival_times) / len(inter_arrival_times)
                max_rtt = max(inter_arrival_times)
        
        # 4. 时间窗口和吞吐量
        if len(packets_list) >= 2:
            first_pkt = packets_list[0]
            last_pkt = packets_list[-1]
            time_window = last_pkt.receive_timestamp - first_pkt.receive_timestamp
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
            arrival_times = [p.receive_timestamp for p in video_packets]
            if len(arrival_times) >= 4:
                inter_arrival_times = [arrival_times[i] - arrival_times[i-1] 
                                      for i in range(1, len(arrival_times))]
                mid = len(inter_arrival_times) // 2
                recent_avg = sum(inter_arrival_times[mid:]) / len(inter_arrival_times[mid:]) if inter_arrival_times[mid:] else 0
                prev_avg = sum(inter_arrival_times[:mid]) / len(inter_arrival_times[:mid]) if inter_arrival_times[:mid] else 0
                if time_window > 0:
                    delay_gradient = (recent_avg - prev_avg) / (time_window / 1000.0)  # ms/s
        
        # 6. 序列号分析（检测丢包）
        seq_nums = [p.sequence_number for p in video_packets if p.sequence_number is not None]
        packet_loss_rate = 0.0
        if len(seq_nums) >= 2:
            min_seq = min(seq_nums)
            max_seq = max(seq_nums)
            expected_packets = max_seq - min_seq + 1
            if expected_packets > 0:
                packet_loss_rate = 1.0 - (len(seq_nums) / expected_packets)
                packet_loss_rate = max(0.0, min(1.0, packet_loss_rate))
        
        # 7. 返回通用特征（适用于所有算法）
        return {
            # 数据包统计
            'total_packets': total_packets,
            'video_packets': video_packets_count,
            'total_bytes': total_bytes,
            'video_bytes': video_bytes,
            
            # RTT/延迟统计
            'min_rtt_ms': min_rtt,
            'avg_rtt_ms': avg_rtt,
            'max_rtt_ms': max_rtt,
            
            # 丢包率
            'loss_rate': packet_loss_rate,
            
            # 时间窗口和吞吐量
            'time_window_ms': time_window,
            'throughput_bps': throughput,
            
            # 延迟梯度
            'delay_gradient_ms_per_s': delay_gradient,
            
            # 当前带宽（通用特征）
            'current_bandwidth_bps': self.current_bitrate,
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
            'current_bandwidth_bps': self.current_bitrate,
        }
    
    def _save_data_record(self, features: dict, bandwidth: int):
        """保存一条数据记录"""
        record = {
            "timestamp": self.call_count,
            "input_features": features,
            "output_bandwidth": bandwidth,
            "algorithm": "BBR"
        }
        self.data_records.append(record)
        self.call_count += 1
    
    def save_data(self, filename: str = None, auto_increment: bool = True):
        """
        保存收集的数据到 JSON 文件
        
        Args:
            filename: 输出文件名，如果为 None 则使用默认名称
            auto_increment: 如果文件已存在，是否自动递增文件名
        
        Returns:
            保存的文件路径，如果未启用数据收集则返回 None
        """
        if not self.enable_data_collection or not self.data_records:
            return None
        
        if filename is None:
            filename = "BBR_imitation_data.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 自动递增文件名
        if auto_increment and os.path.exists(filepath):
            base_name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(filepath):
                filename = f"{base_name}_{counter}{ext}"
                filepath = os.path.join(self.output_dir, filename)
                counter += 1
        
        output_data = {
            "algorithm": "BBR",
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


class WindowedMaxFilter:
    def __init__(self, window_len_ms):
        self.window_len = window_len_ms
        self.samples = collections.deque() # (value, time)

    def update(self, value, now_ms):
        while self.samples and (now_ms - self.samples[0][1] > self.window_len):
            self.samples.popleft()
        self.samples.append((value, now_ms))

    def get_best(self):
        if not self.samples:
            return 0
        return max(s[0] for s in self.samples)

class PacketInfo:
    def __init__(self):
        self.payload_type = None
        self.sequence_number = None
        self.send_timestamp = None
        self.receive_timestamp = None
        self.size = None