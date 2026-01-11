import subprocess
import json
from collections import defaultdict
import time

from utils.ssim import calculate_video_ssim
from evaluate.utils.net_info import NetInfo
from evaluate.utils.net_eval_method import NetEvalMethodExtension
from evaluate.eval_network import init_network_argparse, get_network_score
from utils.draw import *

TRACE_FILES = {
    # 原有的 trace 文件（已注释）
    #'att16': ["ATT-LTE-driving-2016.down", "ATT-LTE-driving-2016.down"],
    #'taxi': ["trace-1552767958-taxi1", "trace-1552767958-taxi1"],
    #'verizon': ["Verizon-LTE-driving.down", "Verizon-LTE-driving.down"],
    #'tmobile': ["TMobile-LTE-driving.down", "TMobile-LTE-driving.down"],
    #'30mbps': ["med_30mbps.trace", "med_30mbps.trace"]
    
    # 新转换的 trace 文件
    '4G_3mbps': ["4G_3mbps.up", "4G_3mbps.up"],
    '4G_500kbps': ["4G_500kbps.up", "4G_500kbps.up"],
    '4G_700kbps': ["4G_700kbps.up", "4G_700kbps.up"],
    '5G_12mbps': ["5G_12mbps.up", "5G_12mbps.up"],
    '5G_13mbps': ["5G_13mbps.up", "5G_13mbps.up"],
    'WIRED_200kbps': ["WIRED_200kbps.up", "WIRED_200kbps.up"],
    'WIRED_35mbps': ["WIRED_35mbps.up", "WIRED_35mbps.up"],
    'WIRED_900kbps': ["WIRED_900kbps.up", "WIRED_900kbps.up"],
    'trace_300k': ["trace_300k.up", "trace_300k.up"],
    'trace_example': ["trace_example.up", "trace_example.up"],
}

# 网络配置：delay(ms)单向延迟, queue为队列大小（数据包数量）
# 队列设计原则：
# - 低带宽场景：队列为BDP的10-15倍，允许拥塞控制探测
# - 高带宽场景：队列为BDP的2-4倍，平衡吞吐量与延迟
# - 长RTT场景：需要更大队列应对慢启动和探测
TRACE_CONFIGS = {
    # 有线网络 (RTT=10ms)
    # BDP 计算：Bandwidth(bps) × RTT(s) / Packet_Size(bits)
    # 队列设为 BDP 的 10-15 倍以应对探测和突发流量
    'WIRED_200kbps': {'delay': 5, 'queue': 15},     # BDP≈0.17包 → 15包 (允许探测)
    'WIRED_900kbps': {'delay': 5, 'queue': 20},     # BDP≈0.75包 → 20包 (允许探测)  
    'WIRED_35mbps': {'delay': 5, 'queue': 70},      # BDP≈29包 → 70包 (2-3倍BDP)
    
    # 4G 网络 (RTT=40ms)
    # 长 RTT 需要更大队列来应对慢启动和探测
    '4G_500kbps': {'delay': 20, 'queue': 25},       # BDP≈1.67包 → 25包 (15倍BDP)
    '4G_700kbps': {'delay': 20, 'queue': 30},       # BDP≈2.33包 → 30包 (13倍BDP)
    '4G_3mbps': {'delay': 20, 'queue': 40},         # BDP≈10包 → 40包 (4倍BDP)
    
    # 5G 网络 (RTT=20ms)
    # 高带宽场景，队列为 2.5-3 倍 BDP
    '5G_12mbps': {'delay': 10, 'queue': 60},        # BDP≈20包 → 60包 (3倍BDP)
    '5G_13mbps': {'delay': 10, 'queue': 65},        # BDP≈21.67包 → 65包 (3倍BDP)
    
    # 其他场景 (长 RTT + 低带宽，需要大队列)
    'trace_300k': {'delay': 50, 'queue': 35},       # BDP≈2.5包 → 35包 (14倍BDP，RTT=100ms)
    'trace_example': {'delay': 45, 'queue': 40},    # BDP≈3包 → 40包 (13倍BDP，RTT=90ms)
}

RESULTS = defaultdict(dict) # key: trace, value: dict of results
ALGORITHMS = [
            #"dummy", 
            "HRCC", 
            "GCC",
            #"Cubic",
            #"PCC",
            #"Copa",
            #"Copa+",
            "BBR",
            "Gemini",
            "FARC",
            "Schaferct",
            ] 
N_TRACES = len(TRACE_FILES)
N_ALGORITHMS = len(ALGORITHMS)


def configure_env_file(algorithm: str, debug=False):
    try:
        with open(".env", "w", encoding='utf-8') as envf:
            envf.write("# .env")
            envf.write(f"\nARG_A=\"{algorithm}\"")
            # 调试模式配置（默认关闭）
            if debug:
                envf.write(f"\nFARC_DEBUG=1")
                envf.write(f"\nSCHAFERCT_DEBUG=1")
            else:
                envf.write(f"\nFARC_DEBUG=0")
                envf.write(f"\nSCHAFERCT_DEBUG=0")
    except Exception as e:
        print(f"Error: {e}")
        raise

def configure_mahimahi_trace(trace: str):
    """
    配置mahimahi的trace、delay和queue参数
    
    Args:
        trace: trace名称
    """
    try:
        # 获取trace配置
        config = TRACE_CONFIGS.get(trace, {'delay': 25, 'queue': 30})
        
        # 读取并更新配置文件
        with open("share/input/cases/trace/mahimahi.json", "r", encoding='utf-8') as tracef:
            trace_data = json.load(tracef)
        
        # 更新配置
        trace_data["link"] = TRACE_FILES[trace]
        trace_data["delay"] = config['delay']  # 单向延迟（ms）
        trace_data["queue"] = config['queue']  # 队列大小（数据包数量）
        
        # 保存配置
        with open("share/input/cases/trace/mahimahi.json", "w", encoding='utf-8') as tracef:
            json.dump(trace_data, tracef, indent=2)
        
        print(f"📝 配置 {trace}: delay={config['delay']}ms (RTT={2*config['delay']}ms), queue={config['queue']}包")
    
    except Exception as e:
        print(f"Error: {e}")
        raise

def run_one_scenario(algorithm: str, trace: str, timeout=65):
    """
    运行单个测试场景
    
    Args:
        algorithm: 算法名称
        trace: trace 名称
        timeout: 超时时间（秒），默认 65 秒（根据 autoclose=60 + 5秒缓冲）
    """
    configure_env_file(algorithm)
    configure_mahimahi_trace(trace)

    # 1. 后台启动容器
    command = ["docker", "compose", "up", "-d"]
    print(f"Executing: {command} with algorithm: {algorithm}")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. 等待容器完成任务（带超时保护）
    print(f"Waiting for {algorithm} to complete (timeout: {timeout}s)...")
    command = ["docker", "compose", "wait", "receiver", "sender"]
    
    try:
        subprocess.run(command, check=False, timeout=timeout)
        print(f"✅ {algorithm} completed successfully")
    except subprocess.TimeoutExpired:
        print(f"⚠️ {algorithm} 运行超时 ({timeout}秒)，强制停止容器")
    
    # 3. 停止并清理（无论成功或超时都会执行）
    time.sleep(3)
    command = ["docker", "compose", "down"]
    print(f"Finished: {algorithm}")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def calculate_video_frame_loss_rate(net_parser: NetInfo):
    """
    计算视频帧的丢包率（参考 GCC 的 caculate_loss_rate 方法）
    仅针对视频数据包（payload_type == 125）进行统计
    
    Args:
        net_parser: NetInfo 对象，包含解析后的网络日志数据
    
    Returns:
        video_loss_rate: 视频帧丢包率 (0.0-1.0)，如果无法计算则返回 -1
    """
    net_data = net_parser.net_data
    
    if not net_data or len(net_data) == 0:
        return -1
    
    # 统计视频数据包（payload_type == 125）
    video_packets = []
    for item in net_data:
        try:
            payload_type = item["packetInfo"]["header"]["payloadType"]
            sequence_number = item["packetInfo"]["header"]["sequenceNumber"]
            
            if payload_type == 125:  # 视频数据包
                video_packets.append({
                    "seq": sequence_number,
                    "timestamp": item["packetInfo"]["arrivalTimeMs"]
                })
        except (KeyError, TypeError):
            continue
    
    if len(video_packets) == 0:
        return -1
    
    # 找到序列号范围
    min_seq = video_packets[0]["seq"]
    max_seq = video_packets[0]["seq"]
    
    for pkt in video_packets:
        min_seq = min(min_seq, pkt["seq"])
        max_seq = max(max_seq, pkt["seq"])
    
    # 序列号范围为 0 说明只有一个包，无法计算丢包率
    if (max_seq - min_seq) == 0:
        return -1
    
    # 计算接收率和丢包率
    # 理论上应该收到的包数：(max_seq - min_seq + 1)
    # 实际收到的包数：len(video_packets)
    expected_packets = max_seq - min_seq + 1
    received_packets = len(video_packets)
    
    # 接收率 = 实际收到 / 理论应该收到
    receive_rate = received_packets / expected_packets
    # 丢包率 = 1 - 接收率
    video_loss_rate = 1 - receive_rate
    
    return video_loss_rate


def evaluate_one_scenario(trace: str, run_idx: int):
    def network_score(algorithm: str):
        try:
            network_parser = init_network_argparse()
            network_args = network_parser.parse_args()
            network_args.dst_network_log = f"share/output/trace/webrtc_{algorithm}.log"
            return get_network_score(network_args)
        except Exception as e:
            print(f"⚠️ {algorithm} 网络评分失败: {e}")
            return 0.0

    # 动态创建 NetInfo 对象并解析日志
    net_parsers = []
    for alg in ALGORITHMS:
        try:
            net_parser = NetInfo(f'share/output/trace/webrtc_{alg}.log')
            net_parser.parse_net_log()
            net_parsers.append(net_parser)
        except Exception as e:
            print(f"⚠️ {alg} 日志解析失败: {e}")
            # 创建一个空的 NetInfo 对象作为占位符
            net_parser = NetInfo(f'share/output/trace/webrtc_{alg}.log')
            net_parser.net_data = []  # 空数据
            net_parsers.append(net_parser)

    # 动态评估结果
    net_eval_extension = NetEvalMethodExtension()
    results = []
    for parser in net_parsers:
        try:
            result = net_eval_extension.eval(parser)
            results.append(result)
        except Exception as e:
            print(f"⚠️ 评估失败: {e}")
            # 返回默认的空结果 (时间序列数据, delay1, delay2, goodput, loss, freeze_rate)
            results.append(({}, 0.0, 0.0, 0.0, 0.0, 0.0))
    
    # 创建算法标签（首字母大写）
    alg_labels = [alg.capitalize() if alg.lower() == alg else alg for alg in ALGORITHMS]
    
    if run_idx == 0:
        # 提取 goodput 数据并绘图
        try:
            goodput_data = [result[0] for result in results]
            draw_goodput(goodput_data, alg_labels, f"goodput_time_{trace}")
        except Exception as e:
            print(f"⚠️ 绘制 goodput 图失败: {e}")
        
        # 初始化结果字典
        for alg in ALGORITHMS:
            RESULTS[trace][alg] = defaultdict(dict)
            RESULTS[trace][alg]["delay1"] = []
            RESULTS[trace][alg]["delay2"] = []
            RESULTS[trace][alg]["goodput"] = []
            RESULTS[trace][alg]["loss"] = []
            RESULTS[trace][alg]["video_loss"] = []  # 视频帧丢包率
            RESULTS[trace][alg]["freeze_rate"] = []
            RESULTS[trace][alg]["network score"] = []
            RESULTS[trace][alg]["SSIM"] = []
    
    # 追加结果
    for idx, alg in enumerate(ALGORITHMS):
        try:
            RESULTS[trace][alg]["delay1"].append(results[idx][1])
            RESULTS[trace][alg]["delay2"].append(results[idx][2])
            RESULTS[trace][alg]["goodput"].append(results[idx][3])
            RESULTS[trace][alg]["loss"].append(results[idx][4])
            RESULTS[trace][alg]["freeze_rate"].append(results[idx][5])
            RESULTS[trace][alg]["network score"].append(network_score(alg))
            
            # 计算视频帧丢包率（参考 GCC 方法）
            try:
                video_loss = calculate_video_frame_loss_rate(net_parsers[idx])
                RESULTS[trace][alg]["video_loss"].append(video_loss if video_loss != -1 else 0.0)
            except Exception as e:
                print(f"⚠️ {alg} 视频帧丢包率计算失败: {e}")
                RESULTS[trace][alg]["video_loss"].append(0.0)
            
            # SSIM 计算也可能失败（视频文件不存在）
            try:
                ssim_value = calculate_video_ssim(f"share/input/testmedia/test.y4m", f"share/output/trace/outvideo_{alg}.y4m")
                RESULTS[trace][alg]["SSIM"].append(ssim_value)
            except Exception as e:
                print(f"⚠️ {alg} SSIM 计算失败: {e}")
                RESULTS[trace][alg]["SSIM"].append(0.0)
        except Exception as e:
            print(f"⚠️ {alg} 结果追加失败: {e}")
            # 追加默认值
            RESULTS[trace][alg]["delay1"].append(0.0)
            RESULTS[trace][alg]["delay2"].append(0.0)
            RESULTS[trace][alg]["goodput"].append(0.0)
            RESULTS[trace][alg]["loss"].append(0.0)
            RESULTS[trace][alg]["video_loss"].append(0.0)
            RESULTS[trace][alg]["freeze_rate"].append(0.0)
            RESULTS[trace][alg]["network score"].append(0.0)
            RESULTS[trace][alg]["SSIM"].append(0.0)

def demo(times=5, file_name="share/output/trace/demo_results.json"):
    for t_idx, trace in enumerate(list(TRACE_FILES.keys())):
        for i in range(times):
            for alg in ALGORITHMS:
                run_one_scenario(alg, trace)
            print(f"({(t_idx)*times+i+1}/{times*N_TRACES}): Finished {i+1} times of {trace} trace")
            evaluate_one_scenario(trace, i)
        
        # 每个场景完成后立即保存（增量保存）
        with open(file_name, "w", encoding='utf-8') as resf:
            json.dump(RESULTS, resf, indent=2)
        print(f"✅ [{t_idx+1}/{N_TRACES}] {trace} 场景数据已保存到 {file_name}")
    
    print(f"🎉 所有场景完成！最终结果已保存到 {file_name}")

def visual_demo(json_file):
    for alg in ALGORITHMS:
        draw_metrics_from_json_traces(json_file, alg, "delay1", "goodput", ("Self-Inflicted Delay (ms)", "Average Goodput (Mbps)"))
        draw_metrics_from_json_traces(json_file, alg, "delay2", "goodput", ("95th Percentile One-Way Delay (ms)", "Average Goodput (Mbps)"))

    draw_combined_scores_from_json_traces(json_file)

 
if __name__ == '__main__':

    demo(1)
    #visual_demo("share/output/trace/demo_results.json")
