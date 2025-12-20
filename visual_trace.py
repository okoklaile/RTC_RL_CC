import sys
import os

# 将当前目录加入路径，以便导入 utils
sys.path.append(os.getcwd())

from evaluate.utils.net_info import NetInfo
from evaluate.utils.net_eval_method import NetEvalMethodExtension
from utils.draw import draw_goodput

def visual_trace_result(algorithm="dummy"):
    # 1. 确定日志文件路径
    # docker compose 默认输出到 share/output/trace/webrtc_{算法名}.log
    log_path = f"share/output/trace/webrtc_{algorithm}.log"
    
    if not os.path.exists(log_path):
        print(f"错误: 找不到日志文件 {log_path}")
        print("请确认您已经运行过 'docker compose up' 且没有报错。")
        return

    print(f"正在解析日志: {log_path} ...")

    # 2. 解析网络日志
    net_info = NetInfo(log_path)
    
    # 3. 计算 Goodput 数据
    # NetEvalMethodExtension.eval 返回一个元组，第一个元素是 time_nbytes 字典
    eval_method = NetEvalMethodExtension()
    result = eval_method.eval(net_info)
    time_nbytes = result[0]

    # 4. 绘图
    # draw_goodput 接受列表形式的参数，因为可以同时画多条流
    output_name = f"goodput_trace_{algorithm}"
    draw_goodput(
        [time_nbytes],           # 数据列表
        [algorithm],             # 标签列表
        output_name,             # 输出文件名 (不带后缀)
        min_gap=200,             # 统计窗口 (ms)
        duration=60              # 绘图时长 (s)，trace 场景默认通常是 60秒
    )
    
    print(f"绘图完成! 请查看: share/output/figures/{output_name}.pdf")

if __name__ == "__main__":
    # 如果您修改了 compose.yaml 中的 ARG_A 环境变量，请修改这里的算法名
    #target_algo = "dummy" 
    target_algo = "GCC"
    # target_algo = "HRCC"
    
    visual_trace_result(target_algo)