#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .net_info import NetInfo
import numpy as np
from abc import ABC, abstractmethod


class NetEvalMethod(ABC):
    @abstractmethod
    def __init__(self):
        self.eval_name = "base"

    @abstractmethod
    def eval(self, dst_audio_info : NetInfo):
        pass


class NetEvalMethodNormal(NetEvalMethod):
    def __init__(self, max_delay=400, ground_recv_rate=500):
        super(NetEvalMethodNormal, self).__init__()
        self.eval_name = "normal"
        self.max_delay = max_delay
        self.ground_recv_rate = ground_recv_rate

    def eval(self, dst_audio_info : NetInfo):
        net_data = dst_audio_info.net_data
        ssrc_info = {}

        delay_list = []
        loss_count = 0
        self.last_seqNo = {}
        for item in net_data:
            ssrc = item["packetInfo"]["header"]["ssrc"]
            sequence_number = item["packetInfo"]["header"]["sequenceNumber"]
            tmp_delay = item["packetInfo"]["arrivalTimeMs"] - item["packetInfo"]["header"]["sendTimestamp"]
            if (ssrc not in ssrc_info):
                ssrc_info[ssrc] = {
                    "time_delta" : -tmp_delay,
                    "delay_list" : [],
                    "received_nbytes" : 0,
                    "start_recv_time" : item["packetInfo"]["arrivalTimeMs"],
                    "avg_recv_rate" : 0
                }
            if ssrc in self.last_seqNo:
                loss_count += max(0, sequence_number - self.last_seqNo[ssrc] - 1)
            self.last_seqNo[ssrc] = sequence_number
                
            ssrc_info[ssrc]["delay_list"].append(ssrc_info[ssrc]["time_delta"] + tmp_delay)
            ssrc_info[ssrc]["received_nbytes"] += item["packetInfo"]["payloadSize"]
            if item["packetInfo"]["arrivalTimeMs"] != ssrc_info[ssrc]["start_recv_time"]:
                ssrc_info[ssrc]["avg_recv_rate"] = ssrc_info[ssrc]["received_nbytes"] / (item["packetInfo"]["arrivalTimeMs"] - ssrc_info[ssrc]["start_recv_time"])
            
        # scale delay list
        for ssrc in ssrc_info:
            min_delay = min(ssrc_info[ssrc]["delay_list"])
            ssrc_info[ssrc]["scale_delay_list"] = [min(self.max_delay, delay) for delay in ssrc_info[ssrc]["delay_list"]]
            delay_pencentile_95 = np.percentile(ssrc_info[ssrc]["scale_delay_list"], 95)
            ssrc_info[ssrc]["delay_score"] = (self.max_delay - delay_pencentile_95) / (self.max_delay - min_delay)
        # delay score
        avg_delay_score = np.mean([np.mean(ssrc_info[ssrc]["delay_score"]) for ssrc in ssrc_info])

        # receive rate score
        recv_rate_list = [ssrc_info[ssrc]["avg_recv_rate"] for ssrc in ssrc_info if ssrc_info[ssrc]["avg_recv_rate"] > 0]
        avg_recv_rate_score = min(1, np.mean(recv_rate_list) / self.ground_recv_rate)

        # higher loss rate, lower score
        avg_loss_rate = loss_count / (loss_count + len(net_data))

        # calculate result score
        network_score = 100 * 0.2 * avg_delay_score + \
                            100 * 0.2 * avg_recv_rate_score + \
                            100 * 0.3 * (1 - avg_loss_rate)

        # Note: NetEvalMethodNormal returns a single score, not tuple like Extension
        return network_score


class NetEvalMethodExtension(NetEvalMethod):
    def __init__(self, max_delay=400, ground_recv_rate=500):
        super(NetEvalMethodExtension, self).__init__()
        self.eval_name = "extension"
        self.max_delay = max_delay
        self.ground_recv_rate = ground_recv_rate
    
    def eval(self, dst_net_info: NetInfo):
        net_data = dst_net_info.net_data
        
        # 处理空数据情况
        if not net_data or len(net_data) == 0:
            print(f"⚠️ net_data 为空，返回默认值")
            return ({}, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        ssrc_info = {}
        time_nbytes = {}

        loss_count = 0
        self.last_seqNo = {}
        for item in net_data:
            ssrc = item["packetInfo"]["header"]["ssrc"]
            sequence_number = item["packetInfo"]["header"]["sequenceNumber"]
            tmp_delay = item["packetInfo"]["arrivalTimeMs"] - item["packetInfo"]["header"]["sendTimestamp"]
            timestamp = item["packetInfo"]["arrivalTimeMs"]
            if (ssrc not in ssrc_info):
                ssrc_info[ssrc] = {
                    "time_delta" : -tmp_delay,
                    "delay_list" : [],
                    "received_nbytes" : 0,
                    "start_recv_time" : item["packetInfo"]["arrivalTimeMs"],
                    "avg_recv_rate" : 0,
                    "packet_cnt": 0
                }
            
            if (timestamp not in time_nbytes):
                time_nbytes[timestamp] = item["packetInfo"]["payloadSize"]
            else:
                time_nbytes[timestamp] += item["packetInfo"]["payloadSize"]

            if ssrc in self.last_seqNo:
                loss_count += max(0, sequence_number - self.last_seqNo[ssrc] - 1)
            self.last_seqNo[ssrc] = sequence_number

            ssrc_info[ssrc]["delay_list"].append(ssrc_info[ssrc]["time_delta"] + tmp_delay)
            ssrc_info[ssrc]["received_nbytes"] += item["packetInfo"]["payloadSize"]
            if item["packetInfo"]["arrivalTimeMs"] != ssrc_info[ssrc]["start_recv_time"]:
                ssrc_info[ssrc]["avg_recv_rate"] = ssrc_info[ssrc]["received_nbytes"] / (item["packetInfo"]["arrivalTimeMs"] - ssrc_info[ssrc]["start_recv_time"])
        
        for ssrc in ssrc_info:
            min_delay = min(ssrc_info[ssrc]["delay_list"])
            ssrc_info[ssrc]["scale_delay_list"] = [min(self.max_delay, delay) for delay in ssrc_info[ssrc]["delay_list"]]
        
        # all_self_inflicted_delays = []
        # all_delay_pencentile_95 = []
        # for ssrc in ssrc_info:
        #     min_delay = min(ssrc_info[ssrc]["scale_delay_list"])
        #     self_inflicted_delay = [delay - min_delay for delay in ssrc_info[ssrc]["scale_delay_list"]]
        #     all_self_inflicted_delays.extend(self_inflicted_delay)
        #     all_delay_pencentile_95.append(np.percentile(ssrc_info[ssrc]["scale_delay_list"], 95))
            
        #     recv_rate_list = [ssrc_info[ssrc]["avg_recv_rate"]*8. /1000. for ssrc in ssrc_info if ssrc_info[ssrc]["avg_recv_rate"] > 0]

        # time = list(time_nbytes.keys())
        # nbytes = list(time_nbytes.values())
        # avg_good_put = (np.sum(nbytes)*8/1000) / (time[-1]-time[0])
        
        all_self_inflicted_delays = []
        all_delay_pencentile_95 = []
        for ssrc in ssrc_info:
            min_delay = min(ssrc_info[ssrc]["delay_list"])
            self_inflicted_delay = [delay - min_delay for delay in ssrc_info[ssrc]["delay_list"]]
            all_self_inflicted_delays.extend(self_inflicted_delay)
            all_delay_pencentile_95.append(np.percentile(ssrc_info[ssrc]["delay_list"], 95))
        
        # 计算接收速率列表（移到循环外部）
        recv_rate_list = [ssrc_info[ssrc]["avg_recv_rate"]*8. /1000. for ssrc in ssrc_info if ssrc_info[ssrc]["avg_recv_rate"] > 0]

        # 防止空列表导致 np.mean 返回 nan
        mean_self_inflicted = np.mean(all_self_inflicted_delays) if all_self_inflicted_delays else 0.0
        mean_delay_95 = np.mean(all_delay_pencentile_95) if all_delay_pencentile_95 else 0.0
        sum_recv_rate = np.sum(recv_rate_list) if recv_rate_list else 0.0
        
        # 防止除零错误
        total_packets = loss_count + len(net_data)
        loss_ratio = loss_count / total_packets if total_packets > 0 else 0.0
        
        # 计算卡顿率（从最后一条有效数据中提取videoInfo）
        freeze_rate = 0.0
        try:
            # 从后往前查找有效的 videoInfo 数据
            for item in reversed(net_data):
                if "mediaInfo" in item and "videoInfo" in item["mediaInfo"]:
                    video_info = item["mediaInfo"]["videoInfo"]
                    frames_dropped = video_info.get("framesDroped", 0)
                    frames_received = video_info.get("framesReceived", 0)
                    
                    # 检查是否是有效数据（不是初始值 18446744073709551615）
                    if frames_received < 18446744073709551615 and frames_received > 0:
                        freeze_rate = frames_dropped / (frames_received + frames_dropped)
                        break
        except Exception as e:
            print(f"⚠️ 计算卡顿率失败: {e}")
            freeze_rate = 0.0
        
        return (time_nbytes, mean_self_inflicted, mean_delay_95, sum_recv_rate, loss_ratio, freeze_rate)