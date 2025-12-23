import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.ssim import calculate_video_ssim
from utils.draw import draw_goodput
from evaluate.utils.net_info import NetInfo
from evaluate.utils.net_eval_method import NetEvalMethodExtension
from evaluate.eval_network import init_network_argparse, get_network_score


def network_score(index:int, algorithm: str, topo: str):
    network_parser = init_network_argparse()
    network_args = network_parser.parse_args()
    network_args.dst_network_log = f"share/output/{topo}/webrtc{index}_{algorithm}.log"
    return get_network_score(network_args)

def visual_one_topo(max_index:int, topo: str):
    for algorithm in ["dummy", "HRCC", "GCC"]:
        net_parsers = []
        results = []
        flow_labels = []
        net_eval_extension = NetEvalMethodExtension()
        for i in range(1, max_index+1):
            log_path = f"share/output/{topo}/webrtc{i}_{algorithm}.log"
            if not os.path.exists(log_path):
                print(f"=> Warning: Log file not found: {log_path}, skipping...")
                continue
            try:
                net_parser = NetInfo(log_path)
                net_parsers.append(net_parser)
                result = net_eval_extension.eval(net_parser)
                results.append(result)
                flow_labels.append(f"Flow {i}")
            except Exception as e:
                print(f"=> Warning: Failed to process {log_path}: {e}, skipping...")
                continue
        
        if results:
            draw_goodput([r[0] for r in results], flow_labels, f"goodput_time_{topo}_{algorithm}", min_gap=200, duration=20)
        else:
            print(f"=> No valid log files found for {topo} with algorithm {algorithm}")
    
    

if __name__ == "__main__":
    visual_one_topo(2, "dumbbell")
    #visual_one_topo(3, "parkinglot")
    #visual_one_topo(1, "one2one")