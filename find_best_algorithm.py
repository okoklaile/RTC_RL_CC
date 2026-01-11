#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
找出每个场景下表现最好的算法
基于多指标加权评分：吞吐高、丢包低、延迟低、SSIM高
"""

import json
import pandas as pd
import numpy as np

# ============ 评分权重配置 ============
# 可以根据需求调整这些系数
WEIGHTS = {
    'goodput': 1.0,      # 吞吐量权重（越高越好）
    'loss': -50.0,       # 丢包率权重（越低越好，负权重）
    'delay95': -0.01,    # 95尾延迟权重（越低越好，负权重）
    'ssim': 100.0        # SSIM视频质量权重（越高越好）
}

def calculate_custom_score(metrics):
    """
    计算自定义评分
    
    公式: score = w1*goodput + w2*(-loss) + w3*(-delay95) + w4*ssim
    
    参数说明:
    - goodput: 吞吐量 (Mbps)，越高越好
    - loss: 丢包率 (0-1)，越低越好
    - delay95: 95分位延迟 (ms)，使用delay2作为95尾延迟，越低越好
    - ssim: 视频质量 (0-1)，越高越好
    """
    goodput = metrics.get('goodput', [0, 0])
    loss = metrics.get('loss', [0, 0])
    delay95 = metrics.get('delay2', [0, 0])  # delay2作为95尾延迟
    ssim = metrics.get('SSIM', [0, 0])
    
    # 计算平均值，过滤0值
    avg_goodput = np.mean([g for g in goodput if g != 0]) if any(goodput) else 0
    avg_loss = np.mean([l for l in loss if l != 0]) if any(loss) else 0
    avg_delay95 = np.mean([d for d in delay95 if d != 0]) if any(delay95) else 0
    avg_ssim = np.mean([s for s in ssim if s != 0]) if any(ssim) else 0
    
    # 计算加权得分
    score = (
        WEIGHTS['goodput'] * avg_goodput +
        WEIGHTS['loss'] * avg_loss +
        WEIGHTS['delay95'] * avg_delay95 +
        WEIGHTS['ssim'] * avg_ssim
    )
    
    return score, {
        'avg_goodput': avg_goodput,
        'avg_loss': avg_loss,
        'avg_delay95': avg_delay95,
        'avg_ssim': avg_ssim
    }

def find_best_algorithms(json_file):
    """分析每个场景下最佳算法"""
    
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = []
    
    # 打印权重配置
    print("\n" + "="*70)
    print("📊 评分权重配置")
    print("="*70)
    print(f"  吞吐量 (goodput)   权重: {WEIGHTS['goodput']:8.2f}  [越高越好]")
    print(f"  丢包率 (loss)      权重: {WEIGHTS['loss']:8.2f}  [越低越好]")
    print(f"  95尾延迟 (delay95) 权重: {WEIGHTS['delay95']:8.2f}  [越低越好]")
    print(f"  视频质量 (SSIM)    权重: {WEIGHTS['ssim']:8.2f}  [越高越好]")
    print("="*70)
    print("💡 提示: 可以在脚本开头的 WEIGHTS 字典中修改这些系数\n")
    
    # 遍历每个场景
    for scenario, algorithms in data.items():
        print(f"\n{'='*60}")
        print(f"场景: {scenario}")
        print(f"{'='*60}")
        
        scenario_scores = {}
        
        # 计算每个算法的自定义评分
        for algo_name, metrics in algorithms.items():
            custom_score, metric_avgs = calculate_custom_score(metrics)
            
            scenario_scores[algo_name] = {
                'custom_score': custom_score,
                'goodput': metrics.get('goodput', [0, 0]),
                'loss': metrics.get('loss', [0, 0]),
                'delay95': metrics.get('delay2', [0, 0]),
                'ssim': metrics.get('SSIM', [0, 0]),
                'avg_goodput': metric_avgs['avg_goodput'],
                'avg_loss': metric_avgs['avg_loss'],
                'avg_delay95': metric_avgs['avg_delay95'],
                'avg_ssim': metric_avgs['avg_ssim']
            }
        
        # 找出最佳算法（基于自定义评分）
        if scenario_scores:
            best_algo = max(scenario_scores.items(), key=lambda x: x[1]['custom_score'])
            best_name = best_algo[0]
            best_info = best_algo[1]
            
            print(f"\n🏆 最佳算法: {best_name}")
            print(f"   自定义评分: {best_info['custom_score']:.2f}")
            print(f"   ├─ 平均吞吐量 (Goodput):  {best_info['avg_goodput']:.3f} Mbps")
            print(f"   ├─ 平均丢包率 (Loss):     {best_info['avg_loss']:.4f} ({best_info['avg_loss']*100:.2f}%)")
            print(f"   ├─ 平均95尾延迟 (Delay): {best_info['avg_delay95']:.2f} ms")
            print(f"   └─ 平均视频质量 (SSIM):   {best_info['avg_ssim']:.4f}")
            
            # 显示所有算法排名
            print(f"\n  所有算法排名:")
            sorted_algos = sorted(scenario_scores.items(), key=lambda x: x[1]['custom_score'], reverse=True)
            for rank, (name, info) in enumerate(sorted_algos, 1):
                print(f"    {rank}. {name:12s} - 得分: {info['custom_score']:8.2f} "
                      f"(吞吐:{info['avg_goodput']:6.3f} 丢包:{info['avg_loss']:.4f} "
                      f"延迟:{info['avg_delay95']:6.1f} SSIM:{info['avg_ssim']:.3f})")
            
            results.append({
                'scenario': scenario,
                'best_algorithm': best_name,
                'custom_score': best_info['custom_score'],
                'goodput_avg': best_info['avg_goodput'],
                'loss_avg': best_info['avg_loss'],
                'delay95_avg': best_info['avg_delay95'],
                'ssim_avg': best_info['avg_ssim']
            })
    
    # 生成总结表格
    print(f"\n\n{'='*100}")
    print("📊 最佳算法总结（基于自定义多指标评分）")
    print(f"{'='*100}\n")
    
    df = pd.DataFrame(results)
    # 格式化输出
    pd.options.display.float_format = '{:.4f}'.format
    print(df.to_string(index=False))
    
    # 保存结果到文件
    output_file = 'best_algorithms_summary_custom.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 结果已保存到: {output_file}")
    
    # 统计各算法获胜次数
    print(f"\n{'='*70}")
    print("🏅 算法获胜统计（基于自定义评分）")
    print(f"{'='*70}")
    algo_wins = {}
    for r in results:
        algo = r['best_algorithm']
        algo_wins[algo] = algo_wins.get(algo, 0) + 1
    
    for algo, count in sorted(algo_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"  {algo:12s}: {count} 个场景")
    
    # 打印权重提示
    print(f"\n{'='*70}")
    print("💡 如需调整评分标准，请修改脚本开头的 WEIGHTS 配置")
    print(f"{'='*70}")
    print("  当前权重:")
    print(f"    WEIGHTS = {{")
    for key, val in WEIGHTS.items():
        print(f"        '{key}': {val},")
    print(f"    }}")
    print(f"{'='*70}")
    
    return results

if __name__ == '__main__':
    json_file = 'share/output/trace/demo_results.json'
    results = find_best_algorithms(json_file)
