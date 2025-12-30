#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRCC (Hybrid Reinforcement learning and rule-based Congestion Control) 带宽估计器
结合GCC启发式算法和PPO强化学习模型的混合带宽估计方案
- 基础层：使用GCC算法进行带宽估计
- 优化层：使用PPO强化学习模型调整GCC的估计结果
"""
from deep_rl.ppo_agent import PPO
import json
import os
import torch
from packet_info import PacketInfo
from packet_record import PacketRecord
from BandwidthEstimator_gcc import GCCEstimator
class Estimator(object):
    """
    混合带宽估计器
    结合GCC基线算法和PPO强化学习模型，实现更智能的带宽估计
    """
    def __init__(self, model_path="./model/pretrained_model.pth", step_time=200, enable_data_collection: bool = False, output_dir: str = "share/output/imitation_data"):
        """
        初始化混合带宽估计器
        Args:
            model_path: 预训练模型路径
            step_time: 时间步长(毫秒)，默认200ms
            enable_data_collection: 是否启用数据收集（用于模仿学习）
            output_dir: 数据输出目录
        """
        self.enable_data_collection = enable_data_collection
        self.output_dir = output_dir
        self.data_records = []
        self.call_count = 0
        
        if self.enable_data_collection:
            os.makedirs(output_dir, exist_ok=True)
        
        # 1. 定义PPO强化学习模型相关参数
        exploration_param = 0.1  # 动作分布的标准差（探索参数）
        K_epochs = 37  # 策略更新的迭代次数
        ppo_clip = 0.1  # PPO的裁剪参数，限制策略更新幅度
        gamma = 0.99  # 折扣因子，用于计算未来奖励的权重
        lr = 3e-5  # Adam优化器的学习率
        betas = (0.9, 0.999)  # Adam优化器的动量参数
        self.state_dim = 6  # 状态维度：接收率、延迟、丢包率、带宽预测、过载距离、上次过载容量
        self.state_length = 10  # 状态历史长度，保留最近10个时间步的状态
        action_dim = 1  # 动作维度：带宽调整系数
        
        # 2. 加载预训练的PPO模型
        self.device = torch.device("cpu")
        self.ppo = PPO(self.state_dim, self.state_length, action_dim, exploration_param, lr, betas, gamma, K_epochs, ppo_clip)
        self.ppo.policy.load_state_dict(torch.load('./share/input/ccalgs/HRCC/hrcc.pth'))
        
        # 初始化数据包记录器（用于统计网络指标）
        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.step_time = step_time  # 时间步长(ms)
        
        # 3. 初始化状态和控制变量
        self.state = torch.zeros((1, self.state_dim, self.state_length))  # 状态张量 [batch, features, time_steps]
        self.time_to_guide = False  # 是否到达RL指导时机
        self.counter = 0  # 时间步计数器
        self.bandwidth_prediction = 300000  # 带宽预测值(bps)，初始300kbps
        
        # 初始化GCC基线估计器
        self.gcc_estimator = GCCEstimator()
        
        # 网络指标历史列表
        self.receiving_rate_list = []  # 接收率历史
        self.delay_list = []  # 延迟历史
        self.loss_ratio_list = []  # 丢包率历史
        self.bandwidth_prediction_list = []  # 带宽预测历史
        
        # 过载检测相关
        self.overuse_flag = 'NORMAL'  # 过载标志：'NORMAL', 'OVERUSE', 'UNDERUSE'
        self.overuse_distance = 5  # 距离上次过载的时间步数
        self.last_overuse_cap = 1000000  # 上次发生过载时的接收率(bps)

    def report_states(self, stats: dict):
        """
        接收并记录数据包信息
        将数据包信息同时传递给packet_record和gcc_estimator进行处理
        
        Args:
            stats: 数据包统计信息字典，包含以下字段：
        {
                "send_time_ms": uint,        # 发送时间戳(毫秒)
                "arrival_time_ms": uint,     # 到达时间戳(毫秒)
                "payload_type": int,         # 载荷类型
                "sequence_number": uint,     # 序列号
                "ssrc": int,                 # 同步源标识符
                "padding_length": uint,      # 填充长度(字节)
                "header_length": uint,       # 头部长度(字节)
                "payload_size": uint         # 载荷大小(字节)
        }
        """
        # 构造PacketInfo对象
        packet_info = PacketInfo()
        packet_info.payload_type = stats["payload_type"]
        packet_info.ssrc = stats["ssrc"]
        packet_info.sequence_number = stats["sequence_number"]
        packet_info.send_timestamp = stats["send_time_ms"]
        packet_info.receive_timestamp = stats["arrival_time_ms"]
        packet_info.padding_length = stats["padding_length"]
        packet_info.header_length = stats["header_length"]
        packet_info.payload_size = stats["payload_size"]
        packet_info.bandwidth_prediction = self.bandwidth_prediction

        # 更新packet_record用于统计网络指标
        self.packet_record.on_receive(packet_info)
        # 更新gcc_estimator用于基线带宽估计
        self.gcc_estimator.report_states(stats)

    def get_estimated_bandwidth(self)->int:
        """
        计算并返回最终的带宽估计值
        混合方案：每4个时间步使用一次RL调整，其余时间使用GCC估计
        
        工作流程：
        1. 计算当前网络状态（接收率、延迟、丢包率等）
        2. 获取GCC的基线带宽估计
        3. 更新状态张量
        4. 每4步使用PPO模型调整GCC估计（其余时间直接使用GCC估计）
        
        Returns:
            bandwidth_prediction: 最终的带宽预测值(bps)
        """
        # 1. 计算当前时间窗口的网络状态指标
        # 计算接收率(bps)
        self.receiving_rate = self.packet_record.calculate_receiving_rate(interval=self.step_time)
        self.receiving_rate_list.append(self.receiving_rate)
        
        # 计算平均延迟(ms)
        self.delay = self.packet_record.calculate_average_delay(interval=self.step_time)
        self.delay_list.append(self.delay)

        # 计算丢包率(0.0-1.0)
        self.loss_ratio = self.packet_record.calculate_loss_ratio(interval=self.step_time)
        self.loss_ratio_list.append(self.loss_ratio)

        # 获取GCC估计器的带宽估计和过载状态
        self.gcc_decision, self.overuse_flag = self.gcc_estimator.get_estimated_bandwidth()
        
        # 更新过载距离和上次过载容量
        if self.overuse_flag == 'OVERUSE':
            self.overuse_distance = 0  # 刚发生过载，距离重置为0
            self.last_overuse_cap = self.receiving_rate  # 记录过载时的接收率
        else:
            self.overuse_distance += 1  # 距离上次过载又过了一步
        
        # 更新状态张量（滑动窗口）
        self.state = self.state.clone().detach()
        self.state = torch.roll(self.state, -1, dims=-1)  # 向左滚动，丢弃最旧的状态

        # 填充最新的状态（归一化到[0,1]范围）
        self.state[0, 0, -1] = self.receiving_rate / 6000000.0  # 接收率归一化（假设最大6Mbps）
        self.state[0, 1, -1] = self.delay / 1000.0  # 延迟归一化（假设最大1000ms）
        self.state[0, 2, -1] = self.loss_ratio  # 丢包率已经在[0,1]范围
        self.state[0, 3, -1] = self.bandwidth_prediction / 6000000.0  # 带宽预测归一化
        self.state[0, 4, -1] = self.overuse_distance / 100.0  # 过载距离归一化
        self.state[0, 5, -1] = self.last_overuse_cap / 6000000.0  # 上次过载容量归一化

        # 维护历史列表长度
        if len(self.receiving_rate_list) == self.state_length:
            self.receiving_rate_list.pop(0)
            self.delay_list.pop(0)
            self.loss_ratio_list.pop(0)

        # 更新计数器
        self.counter += 1
        
        # 每4步触发一次RL指导
        if self.counter % 4 == 0:
            self.time_to_guide = True
            self.counter = 0

        # 2. 使用RL智能体调整GCC的带宽估计
        if self.time_to_guide == True:
            # 使用PPO策略网络预测动作
            action, _, _, _ = self.ppo.policy.forward(self.state)
            # action范围约为[0,1]，映射到调整系数 2^(2*action-1)，范围约为[0.5, 2]
            # 这样可以实现对GCC估计的缩放调整
            self.bandwidth_prediction = self.gcc_decision * pow(2, (2 * action - 1))
            # 更新GCC估计器的带宽值，保持一致性
            self.gcc_estimator.change_bandwidth_estimation(self.bandwidth_prediction)
            self.time_to_guide = False
        else:
            # 非指导时间步，直接使用GCC估计
            self.bandwidth_prediction = self.gcc_decision


        bandwidth = int(self.bandwidth_prediction)
        
        # 数据收集：HRCC 没有 packets_list，使用默认特征
        if self.enable_data_collection:
            features = self._get_default_features()
            self._save_data_record(features, bandwidth)
        
        return bandwidth

    def _extract_features_from_packets(self, packets_list) -> dict:
                """从数据包列表中提取通用特征（适用于所有算法）"""
                if not packets_list:
                    return self._get_default_features()
        
                video_packets = [p for p in packets_list if hasattr(p, 'payload_type') and p.payload_type == 125]
        
                total_packets = len(packets_list)
                video_packets_count = len(video_packets)
                total_bytes = sum(p.size for p in packets_list if hasattr(p, 'size'))
                video_bytes = sum(p.size for p in video_packets if hasattr(p, 'size'))
        
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
        
                if min_rtt == 0 and len(video_packets) >= 2:
                    arrival_times = [p.receive_timestamp for p in video_packets if hasattr(p, 'receive_timestamp')]
                    arrival_times = [t for t in arrival_times if t > 0]
                    if len(arrival_times) >= 2:
                        inter_arrival_times = [arrival_times[i] - arrival_times[i-1] for i in range(1, len(arrival_times))]
                        if inter_arrival_times:
                            min_rtt = min(inter_arrival_times)
                            avg_rtt = sum(inter_arrival_times) / len(inter_arrival_times)
                            max_rtt = max(inter_arrival_times)
        
                if len(packets_list) >= 2:
                    first_pkt = packets_list[0]
                    last_pkt = packets_list[-1]
                    if hasattr(first_pkt, 'receive_timestamp') and hasattr(last_pkt, 'receive_timestamp'):
                        time_window = last_pkt.receive_timestamp - first_pkt.receive_timestamp
                        throughput = (total_bytes * 8 * 1000) / time_window if time_window > 0 else 0
                    else:
                        time_window = 0
                        throughput = 0
                else:
                    time_window = 0
                    throughput = 0
        
                delay_gradient = 0.0
                if len(video_packets) >= 4 and time_window > 0:
                    arrival_times = [p.receive_timestamp for p in video_packets if hasattr(p, 'receive_timestamp')]
                    arrival_times = [t for t in arrival_times if t > 0]
                    if len(arrival_times) >= 4:
                        inter_arrival_times = [arrival_times[i] - arrival_times[i-1] for i in range(1, len(arrival_times))]
                        mid = len(inter_arrival_times) // 2
                        recent_avg = sum(inter_arrival_times[mid:]) / len(inter_arrival_times[mid:]) if inter_arrival_times[mid:] else 0
                        prev_avg = sum(inter_arrival_times[:mid]) / len(inter_arrival_times[:mid]) if inter_arrival_times[:mid] else 0
                        if time_window > 0:
                            delay_gradient = (recent_avg - prev_avg) / (time_window / 1000.0)
        
                seq_nums = [p.sequence_number for p in video_packets if hasattr(p, 'sequence_number') and p.sequence_number is not None]
                packet_loss_rate = 0.0
                if len(seq_nums) >= 2:
                    min_seq = min(seq_nums)
                    max_seq = max(seq_nums)
                    expected_packets = max_seq - min_seq + 1
                    if expected_packets > 0:
                        packet_loss_rate = 1.0 - (len(seq_nums) / expected_packets)
                        packet_loss_rate = max(0.0, min(1.0, packet_loss_rate))
        
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
                    'current_bandwidth_bps': self.bandwidth_prediction,
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
                    'current_bandwidth_bps': self.bandwidth_prediction,
                }
    
    def _save_data_record(self, features: dict, bandwidth: int):
                """保存一条数据记录"""
                record = {
                    "timestamp": self.call_count,
                    "input_features": features,
                    "output_bandwidth": bandwidth,
                    "algorithm": "HRCC"
                }
                self.data_records.append(record)
                self.call_count += 1
    
    def save_data(self, filename: str = None, auto_increment: bool = True):
                """保存收集的数据到 JSON 文件"""
                if not self.enable_data_collection or not self.data_records:
                    return None
        
                if filename is None:
                    filename = "HRCC_imitation_data.json"
        
                filepath = os.path.join(self.output_dir, filename)
        
                if auto_increment and os.path.exists(filepath):
                    base_name, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(filepath):
                        filename = f"{base_name}_{counter}{ext}"
                        filepath = os.path.join(self.output_dir, filename)
                        counter += 1
        
                output_data = {
                    "algorithm": "HRCC",
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
