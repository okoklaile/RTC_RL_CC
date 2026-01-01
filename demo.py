import sys
import argparse
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
    #'att16': ["ATT-LTE-driving-2016.down", "ATT-LTE-driving-2016.down"],
    #'taxi': ["trace-1552767958-taxi1", "trace-1552767958-taxi1"],
    #'verizon': ["Verizon-LTE-driving.down", "Verizon-LTE-driving.down"],
    #'tmobile': ["TMobile-LTE-driving.down", "TMobile-LTE-driving.down"],
    '30mbps': ["med_30mbps.trace", "med_30mbps.trace"]
}
RESULTS = defaultdict(dict) # key: trace, value: dict of results
ALGORITHMS = [
            #"dummy", 
            "HRCC", 
            "GCC",
            #"Cubic",
            "PCC",
            #"Copa",
            #"Copa+",
            #"BBR",
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
            # 为 FARC 算法启用调试模式
            if algorithm == "FARC" or debug:
                envf.write(f"\nFARC_DEBUG=1")
            else:
                envf.write(f"\nFARC_DEBUG=0")
            
            # Schaferct 调试配置
            if algorithm == "Schaferct" or debug:
                envf.write(f"\nSCHAFERCT_DEBUG=1")
            else:
                envf.write(f"\nSCHAFERCT_DEBUG=0")
    except Exception as e:
        print(f"Error: {e}")
        raise

def configure_mahimahi_trace(tarce: str):
    try:
        with open("share/input/cases/trace/mahimahi.json", "r", encoding='utf-8') as tracef:
            trace_data = json.load(tracef)
        trace_data["link"] = TRACE_FILES[tarce]
        with open("share/input/cases/trace/mahimahi.json", "w", encoding='utf-8') as tracef:
            json.dump(trace_data, tracef)
    
    except Exception as e:
        print(f"Error: {e}")
        raise

def run_one_scenario(algorithm: str, trace: str):
    configure_env_file(algorithm)
    configure_mahimahi_trace(trace)

    command = ["docker", "compose", "up"]
    print(f"Executing: {command} with algorithm: {algorithm}")
    
    # 创建调试日志文件
    debug_log_file = f"share/output/trace/{algorithm}_docker_debug.log"
    print(f"调试日志将保存到: {debug_log_file}")
    
    with open(debug_log_file, "w") as log_f:
        # 保留 stderr 以捕获调试信息，stdout 仍然重定向
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=log_f)
    
    time.sleep(3)
    command = ["docker", "compose", "down"]
    print(f"Executing: {command} with algorithm: {algorithm}")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def evaluate_one_scenario(trace: str, run_idx: int):
    def network_score(algorithm: str):
        network_parser = init_network_argparse()
        network_args = network_parser.parse_args()
        network_args.dst_network_log = f"share/output/trace/webrtc_{algorithm}.log"
        return get_network_score(network_args)

    # 动态创建 NetInfo 对象并解析日志
    net_parsers = []
    for alg in ALGORITHMS:
        net_parser = NetInfo(f'share/output/trace/webrtc_{alg}.log')
        net_parser.parse_net_log()
        net_parsers.append(net_parser)

    # 动态评估结果
    net_eval_extension = NetEvalMethodExtension()
    results = [net_eval_extension.eval(parser) for parser in net_parsers]
    
    # 创建算法标签（首字母大写）
    alg_labels = [alg.capitalize() if alg.lower() == alg else alg for alg in ALGORITHMS]
    
    if run_idx == 0:
        # 提取 goodput 数据并绘图
        goodput_data = [result[0] for result in results]
        draw_goodput(goodput_data, alg_labels, f"goodput_time_{trace}")
        # 初始化结果字典
        for alg in ALGORITHMS:
            RESULTS[trace][alg] = defaultdict(dict)
            RESULTS[trace][alg]["delay1"] = []
            RESULTS[trace][alg]["delay2"] = []
            RESULTS[trace][alg]["goodput"] = []
            RESULTS[trace][alg]["loss"] = []
            RESULTS[trace][alg]["network score"] = []
            RESULTS[trace][alg]["SSIM"] = []
    
    # 追加结果
    for idx, alg in enumerate(ALGORITHMS):
        RESULTS[trace][alg]["delay1"].append(results[idx][1])
        RESULTS[trace][alg]["delay2"].append(results[idx][2])
        RESULTS[trace][alg]["goodput"].append(results[idx][3])
        RESULTS[trace][alg]["loss"].append(results[idx][4])
        RESULTS[trace][alg]["network score"].append(network_score(alg))
        RESULTS[trace][alg]["SSIM"].append(calculate_video_ssim(f"share/input/testmedia/test.y4m", f"share/output/trace/outvideo_{alg}.y4m"))

def demo(times=5, file_name="share/output/trace/demo_results.json"):
    for t_idx, trace in enumerate(list(TRACE_FILES.keys())):
        for i in range(times):
            for alg in ALGORITHMS:
                run_one_scenario(alg, trace)
            print(f"({(t_idx)*times+i+1}/{times*N_TRACES}): Finished {i+1} times of {trace} trace")
            evaluate_one_scenario(trace, i)
    with open(file_name, "w", encoding='utf-8') as resf:
        json.dump(RESULTS, resf)

def visual_demo(json_file):
    for alg in ALGORITHMS:
        draw_metrics_from_json_traces(json_file, alg, "delay1", "goodput", ("Self-Inflicted Delay (ms)", "Average Goodput (Mbps)"))
        draw_metrics_from_json_traces(json_file, alg, "delay2", "goodput", ("95th Percentile One-Way Delay (ms)", "Average Goodput (Mbps)"))

    draw_combined_scores_from_json_traces(json_file)

 
if __name__ == '__main__':

    demo(1)
    #visual_demo("share/output/trace/demo_results.json")
