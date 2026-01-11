#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速对比不同权重配置下的最佳算法选择
用于调优权重参数
"""

import json
import numpy as np
from collections import defaultdict

# 定义多组权重配置进行对比
WEIGHT_CONFIGS = {
    "默认均衡": {
        'goodput': 1.0,
        'loss': -50.0,
        'delay95': -0.01,
        'ssim': 100.0
    },
    "低延迟优先": {
        'goodput': 0.5,
        'loss': -50.0,
        'delay95': -0.05,
        'ssim': 80.0
    },
    "高画质优先": {
        'goodput': 1.5,
        'loss': -30.0,
        'delay95': -0.005,
        'ssim': 150.0
    },
    "零丢包优先": {
        'goodput': 0.8,
        'loss': -100.0,
        'delay95': -0.01,
        'ssim': 100.0
    },
    "高吞吐优先": {
        'goodput': 2.0,
        'loss': -30.0,
        'delay95': -0.005,
        'ssim': 80.0
    }
}

def calculate_score(metrics, weights):
    """计算评分"""
    goodput = metrics.get('goodput', [0, 0])
    loss = metrics.get('loss', [0, 0])
    delay95 = metrics.get('delay2', [0, 0])
    ssim = metrics.get('SSIM', [0, 0])
    
    avg_goodput = np.mean([g for g in goodput if g != 0]) if any(goodput) else 0
    avg_loss = np.mean([l for l in loss if l != 0]) if any(loss) else 0
    avg_delay95 = np.mean([d for d in delay95 if d != 0]) if any(delay95) else 0
    avg_ssim = np.mean([s for s in ssim if s != 0]) if any(ssim) else 0
    
    score = (
        weights['goodput'] * avg_goodput +
        weights['loss'] * avg_loss +
        weights['delay95'] * avg_delay95 +
        weights['ssim'] * avg_ssim
    )
    
    return score

def compare_weights(json_file):
    """对比不同权重配置"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*100)
    print("🔬 不同权重配置下的最佳算法对比分析")
    print("="*100)
    
    # 存储每个配置的结果
    all_results = {}
    
    for config_name, weights in WEIGHT_CONFIGS.items():
        print(f"\n{'─'*100}")
        print(f"📊 配置: {config_name}")
        print(f"{'─'*100}")
        print(f"权重: goodput={weights['goodput']}, loss={weights['loss']}, "
              f"delay95={weights['delay95']}, ssim={weights['ssim']}")
        print()
        
        results = []
        
        for scenario, algorithms in data.items():
            scenario_scores = {}
            
            for algo_name, metrics in algorithms.items():
                score = calculate_score(metrics, weights)
                scenario_scores[algo_name] = score
            
            best_algo = max(scenario_scores.items(), key=lambda x: x[1])
            results.append({
                'scenario': scenario,
                'best_algorithm': best_algo[0],
                'score': best_algo[1]
            })
        
        all_results[config_name] = results
        
        # 统计算法获胜次数
        algo_wins = defaultdict(int)
        for r in results:
            algo_wins[r['best_algorithm']] += 1
        
        print("  算法获胜统计:")
        for algo, count in sorted(algo_wins.items(), key=lambda x: x[1], reverse=True):
            print(f"    {algo:12s}: {count} 个场景")
    
    # 生成对比表格
    print(f"\n\n{'='*100}")
    print("📈 各配置下最佳算法对比表")
    print("="*100)
    print()
    
    scenarios = [r['scenario'] for r in all_results[list(WEIGHT_CONFIGS.keys())[0]]]
    
    # 表头
    header = f"{'场景':<20s}"
    for config_name in WEIGHT_CONFIGS.keys():
        header += f" | {config_name:<15s}"
    print(header)
    print("-" * len(header))
    
    # 每个场景的结果
    for i, scenario in enumerate(scenarios):
        row = f"{scenario:<20s}"
        for config_name in WEIGHT_CONFIGS.keys():
            best_algo = all_results[config_name][i]['best_algorithm']
            row += f" | {best_algo:<15s}"
        print(row)
    
    # 算法获胜次数汇总
    print(f"\n{'='*100}")
    print("🏆 各配置下算法获胜次数汇总")
    print("="*100)
    print()
    
    all_algos = set()
    for config_results in all_results.values():
        for r in config_results:
            all_algos.add(r['best_algorithm'])
    
    header = f"{'算法':<12s}"
    for config_name in WEIGHT_CONFIGS.keys():
        header += f" | {config_name:<15s}"
    print(header)
    print("-" * len(header))
    
    for algo in sorted(all_algos):
        row = f"{algo:<12s}"
        for config_name in WEIGHT_CONFIGS.keys():
            count = sum(1 for r in all_results[config_name] if r['best_algorithm'] == algo)
            row += f" | {count:^15d}"
        print(row)
    
    # 一致性分析
    print(f"\n{'='*100}")
    print("🔍 一致性分析")
    print("="*100)
    print()
    
    consistency_count = 0
    for i, scenario in enumerate(scenarios):
        algos_for_scenario = set()
        for config_name in WEIGHT_CONFIGS.keys():
            algos_for_scenario.add(all_results[config_name][i]['best_algorithm'])
        
        if len(algos_for_scenario) == 1:
            consistency_count += 1
            print(f"  ✓ {scenario:<20s}: 所有配置一致选择 {list(algos_for_scenario)[0]}")
        else:
            print(f"  ✗ {scenario:<20s}: 不同配置选择 {', '.join(algos_for_scenario)}")
    
    print(f"\n  一致性比例: {consistency_count}/{len(scenarios)} ({consistency_count/len(scenarios)*100:.1f}%)")
    
    print(f"\n{'='*100}")
    print("💡 分析建议")
    print("="*100)
    print("  1. 一致性高的场景说明算法优势明显，不受权重影响")
    print("  2. 一致性低的场景说明需要根据具体需求选择权重配置")
    print("  3. 如果某个算法在多种配置下都表现优异，说明其泛化能力强")
    print("  4. 可以根据你的应用场景（低延迟/高画质/零丢包等）选择对应配置")
    print("="*100)

if __name__ == '__main__':
    compare_weights('share/output/trace/demo_results.json')
