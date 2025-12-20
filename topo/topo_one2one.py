#!/usr/bin/python

from mininet.net import Containernet
from mininet.node import Controller
from mininet.link import TCLink
from time import sleep
import os
import json
import argparse
import threading

# 获取当前脚本所在的项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DumbbellTopo:
    """
    哑铃型拓扑: Sender <-> S1 <-> S2 <-> Receiver
    Trace 限制应用在 S1 <-> S2 之间
    """
    def __init__(self, trace_file):
        self.net = Containernet(controller=Controller, link=TCLink)
        self.trace_file = trace_file
        self.trace_data = None
        self.bottleneck_link = None # 专门存储中间瓶颈链路
        self.update_thread = None
        self.running = False
        
        if not trace_file:
            raise ValueError("Trace file is required")
        
        self.load_trace(trace_file)
        self.build()

    def load_trace(self, trace_file):
        """加载 JSON trace 文件"""
        # 尝试在 data 目录下寻找，或者直接使用绝对路径
        possible_paths = [
            os.path.join(project_root, 'data', trace_file),
            trace_file
        ]
        
        trace_path = None
        for p in possible_paths:
            if os.path.exists(p):
                trace_path = p
                break
        
        if not trace_path:
            raise FileNotFoundError(f"Trace file not found: {trace_file}")
        
        print(f"=> Loading trace file: {trace_path}")
        with open(trace_path, 'r') as f:
            self.trace_data = json.load(f)
        
        if 'uplink' not in self.trace_data or 'trace_pattern' not in self.trace_data['uplink']:
            raise ValueError("Trace file must contain 'uplink.trace_pattern'")
        
        print(f"=> Loaded {len(self.trace_data['uplink']['trace_pattern'])} trace patterns")

    def get_initial_params(self):
        """获取初始链路参数"""
        if not self.trace_data:
            return {'bw': 10, 'delay': '10ms', 'loss': 0}
        
        patterns = self.trace_data['uplink']['trace_pattern']
        if not patterns:
            return {'bw': 10, 'delay': '10ms', 'loss': 0}
        
        p = patterns[0]
        capacity = p.get('capacity', 1000)
        rtt = p.get('rtt', 20)
        loss = p.get('loss', 0)
        
        return {
            'bw': max(0.1, capacity / 1000.0), # Mininet bw is in Mbps
            'delay': f'{rtt/2}ms',             # One-way delay
            'loss': loss
        }

    def update_link_dynamic(self):
        """
        根据 Trace 循环动态更新 S1 <-> S2 的链路参数
        逻辑：死循环遍历 list，设置参数 -> 睡觉 -> 下一个
        """
        if not self.trace_data or not self.bottleneck_link:
            return
        
        patterns = self.trace_data['uplink']['trace_pattern']
        cycle_count = 0
        
        print("=> Dynamic link update thread started.")

        while self.running:
            cycle_count += 1
            # 直接遍历，不需要任何花里胡哨的索引计算
            for i, pattern in enumerate(patterns):
                if not self.running:
                    break
                
                # 1. 获取参数
                duration_ms = pattern.get('duration', 1000)
                capacity = pattern.get('capacity', 1000) # kbps
                rtt = pattern.get('rtt', 20)
                loss = pattern.get('loss', 0)
                
                # 2. 转换参数
                bw_mbps = max(0.1, capacity / 1000.0) 
                delay_str = f'{rtt/2}ms'
                
                try:
                    # 3. 设置链路 (S1 <-> S2 双向)
                    # 即使 build() 已经设置过一次，这里覆盖一下也无所谓，保证逻辑简单
                    self.bottleneck_link.intf1.config(bw=bw_mbps, delay=delay_str, loss=loss)
                    self.bottleneck_link.intf2.config(bw=bw_mbps, delay=delay_str, loss=loss)
                    
                    # 打印日志稍微简化一点，看着清爽
                    print(f"=> [Cycle {cycle_count} - #{i}] Link: {bw_mbps:.1f}Mbps, {delay_str}, Loss {loss}%")
                except Exception as e:
                    print(f"=> Error updating link: {e}")

                # 4. 睡够时间，再进下一个模式
                sleep(duration_ms / 1000.0)

        print("=> Dynamic update thread stopped.")

    def build(self):
        print("=> Building Dumbbell Topology: H1 <-> S1 <-> S2 <-> H2")
        
        self.net.addController('c0')

        # 1. 创建两台交换机
        s1 = self.net.addSwitch('s1')
        s2 = self.net.addSwitch('s2')

        # 2. 创建发送端和接收端
        sender = self.net.addDocker('h1', ip='192.168.5.101', dimage='pyrtc_image:latest', 
                                    volumes=["{}/share:/app/share".format(project_root)])
        receiver = self.net.addDocker('h2', ip='192.168.5.102', dimage='pyrtc_image:latest', 
                                     volumes=["{}/share:/app/share".format(project_root)])

        # 3. 创建边缘链路 (H1 <-> S1 和 S2 <-> H2)
        # 这些是理想链路，无带宽限制 (1Gbps+)，无额外延迟
        print("=> Configuring ideal edge links...")
        self.net.addLink(sender, s1)  # 默认无限制
        self.net.addLink(s2, receiver) # 默认无限制

        # 4. 创建瓶颈链路 (S1 <-> S2)
        # 这是我们要应用 Trace 的链路
        params = self.get_initial_params()
        print(f"=> Configuring Bottleneck Link (S1 <-> S2) with initial params:")
        print(f"   Bandwidth: {params['bw']:.2f} Mbps")
        print(f"   Delay:     {params['delay']}")
        print(f"   Loss:      {params['loss']}%")
        
        # use_htb=True 对于精确限速非常重要
        self.bottleneck_link = self.net.addLink(s1, s2, 
                                                bw=params['bw'], 
                                                delay=params['delay'], 
                                                loss=params['loss'], 
                                                use_htb=True)

    def run(self, alg):
        print("=> Starting network...")
        self.net.start()
        # === 【修复 3】: 关闭网卡卸载，确保 TC 限速生效 ===
        # 这一步对于 0.1Mbps 这种低带宽仿真至关重要
        if self.bottleneck_link:
            intf1 = self.bottleneck_link.intf1.name
            intf2 = self.bottleneck_link.intf2.name
            print(f"=> Disabling Offloading on {intf1} and {intf2}...")
            # 注意：需要宿主机安装 ethtool
            os.system(f"ethtool -K {intf1} gso off tso off gro off")
            os.system(f"ethtool -K {intf2} gso off tso off gro off")
        # =================================================
        sender = self.net.get('h1')
        receiver = self.net.get('h2')
        
        # 开启链路动态更新线程
        self.running = True
        self.update_thread = threading.Thread(target=self.update_link_dynamic, daemon=True)
        self.update_thread.start()
        
        print(f"=> Starting experiment with algorithm: {alg}")
        
        # 启动接收端 (Receiver)
        # 注意：这里 cmd 参数可能需要根据你实际的 run.py 逻辑微调
        receiver.cmd(f'python run.py -C "one2one" -I 1 -A {alg} > /dev/null 2>&1 &')
        sleep(2) # 给接收端一点启动时间
        
        # 启动发送端 (Sender)
        sender.cmd(f'python run.py --sender -C "one2one" -I 1 -A {alg} > /dev/null 2>&1 &')

        print("=> Experiment running. Waiting for sender to finish...")
        sender.cmd('wait') # 等待发送脚本结束

        print("=> Experiment finished. Cleaning up...")
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2)
        
        self.net.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dumbbell Topology with Trace-driven Bottleneck')
    parser.add_argument('--algorithm', '-A', default="dummy", 
                        type=str, help='Bandwidth estimator algorithm')
    parser.add_argument('--trace', '-T', required=True,
                        type=str, help='Trace file path')
    args = parser.parse_args()

    # 确保以 root 权限运行 Mininet
    if os.geteuid() != 0:
        print("Error: Mininet must be run as root!")
        exit(1)

    topo = DumbbellTopo(trace_file=args.trace)
    topo.run(args.algorithm)