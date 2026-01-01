#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 ONNX 模型转换为 PyTorch 模型并保存
"""
import os
import sys
import torch
import numpy as np

def convert_onnx_to_pytorch(onnx_path, output_path):
    """
    将 ONNX 模型转换为 PyTorch 模型并保存
    
    Args:
        onnx_path: ONNX 模型文件路径
        output_path: 输出的 PyTorch 模型文件路径
    """
    try:
        print(f"Loading ONNX model from: {onnx_path}")
        
        # 检查 ONNX 模型文件是否存在
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")
        
        # 导入必要的库
        try:
            import onnx
            from onnx2pytorch import ConvertModel
        except ImportError as e:
            print(f"ERROR: Required library not installed: {e}")
            print("Please install with: pip install onnx onnx2pytorch")
            sys.exit(1)
        
        # 加载 ONNX 模型
        print("Loading ONNX model...")
        onnx_model = onnx.load(onnx_path)
        
        # 验证 ONNX 模型
        try:
            onnx.checker.check_model(onnx_model)
            print("ONNX model is valid!")
        except onnx.checker.ValidationError as e:
            print(f"WARNING: ONNX model validation failed: {e}")
            print("Continuing anyway...")
        
        # 转换模型
        print("Converting ONNX model to PyTorch...")
        pytorch_model = ConvertModel(onnx_model)
        
        # 设置为评估模式
        pytorch_model.eval()
        
        # 移动到 CPU
        device = torch.device("cpu")
        pytorch_model.to(device)
        
        print("Model converted successfully!")
        
        # 检查 ONNX 模型的输入输出信息
        print("\nONNX Model Input/Output Information:")
        for inp in onnx_model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in inp.type.tensor_type.shape.dim]
            print(f"  Input '{inp.name}': shape {shape}")
        for out in onnx_model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in out.type.tensor_type.shape.dim]
            print(f"  Output '{out.name}': shape {shape}")
        
        # 尝试测试模型（如果失败，只给出警告，不中断保存）
        print("\nTesting model with dummy input...")
        test_successful = False
        try:
            # 根据 ONNX 模型的输入规格准备测试数据
            dummy_obs = np.zeros((1, 1, 150), dtype=np.float32)
            dummy_hidden = np.zeros((1, 128), dtype=np.float32)
            dummy_cell = np.zeros((1, 128), dtype=np.float32)
            
            obs_tensor = torch.from_numpy(dummy_obs).to(device)
            hidden_tensor = torch.from_numpy(dummy_hidden).to(device)
            cell_tensor = torch.from_numpy(dummy_cell).to(device)
            
            with torch.no_grad():
                # 尝试不同的输入方式
                try:
                    outputs = pytorch_model(obs_tensor, hidden_tensor, cell_tensor)
                    test_successful = True
                except (TypeError, RuntimeError, IndexError) as e:
                    print(f"  Positional args failed: {e}")
                    try:
                        outputs = pytorch_model({
                            'obs': obs_tensor,
                            'hidden_states': hidden_tensor,
                            'cell_states': cell_tensor
                        })
                        test_successful = True
                    except Exception as e2:
                        print(f"  Dict input also failed: {e2}")
                        try:
                            outputs = pytorch_model(obs_tensor)
                            test_successful = True
                        except Exception as e3:
                            print(f"  Single tensor input also failed: {e3}")
                            raise e3
            
            if test_successful:
                print("  Model test successful!")
                print(f"  Output type: {type(outputs)}")
                if isinstance(outputs, tuple):
                    print(f"  Output tuple length: {len(outputs)}")
                    for i, out in enumerate(outputs):
                        if hasattr(out, 'shape'):
                            print(f"    Output[{i}] shape: {out.shape}")
                else:
                    if hasattr(outputs, 'shape'):
                        print(f"    Output shape: {outputs.shape}")
        except Exception as e:
            print(f"  WARNING: Model test failed: {e}")
            print("  This may be due to onnx2pytorch conversion limitations.")
            print("  The model will still be saved, but you may need to test it in your application.")
            print("  You can try using onnxruntime directly if the converted model doesn't work.")
        
        # 保存模型
        print(f"Saving PyTorch model to: {output_path}")
        # 如果输出路径有目录，确保目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存整个模型（推荐方式，因为转换后的模型可能不是标准的 PyTorch 模型）
        torch.save({
            'model': pytorch_model,
            'model_state_dict': pytorch_model.state_dict(),
            'input_shape': (1, 1, 150),
            'hidden_size': 128,
        }, output_path)
        
        print(f"PyTorch model saved successfully to: {output_path}")
        print(f"Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        return pytorch_model
        
    except Exception as e:
        print(f"ERROR: Failed to convert model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # 获取脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 默认路径
    onnx_path = os.path.join(current_dir, "fast_and_furious_model.onnx")
    output_path = os.path.join(current_dir, "farc.pth")
    
    # 如果提供了命令行参数，使用命令行参数
    if len(sys.argv) > 1:
        onnx_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    print("=" * 60)
    print("ONNX to PyTorch Model Converter")
    print("=" * 60)
    print(f"Input ONNX model: {onnx_path}")
    print(f"Output PyTorch model: {output_path}")
    print("=" * 60)
    
    convert_onnx_to_pytorch(onnx_path, output_path)
    
    print("=" * 60)
    print("Conversion completed successfully!")
    print("=" * 60)

