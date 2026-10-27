import pickle
import numpy as np
import sys

def show_pkl_full(pkl_path):
    """
    完整展示pkl文件的所有内容
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"=== {pkl_path} 完整数据结构 ===")
    print(f"所有keys: {list(data.keys())}")
    print("=" * 60)
    
    # 遍历所有键值对
    for key in data.keys():
        value = data[key]
        print(f"\n--- {key} ---")
        
        # 如果是numpy数组
        if isinstance(value, np.ndarray):
            print(f"类型: numpy.ndarray")
            print(f"shape: {value.shape}")
            print(f"dtype: {value.dtype}")
            if value.size > 0:
                if value.ndim == 1 and value.shape[0] < 50:
                    # 一维小数组，直接打印
                    print(f"值: {value}")
                elif value.ndim >= 2:
                    # 多维数组，打印前3帧
                    print(f"前3帧:\n{value[:3]}")
                else:
                    print(f"前5个元素: {value[:5]}")
        
        # 如果是列表
        elif isinstance(value, list):
            print(f"类型: list")
            print(f"长度: {len(value)}")
            if len(value) > 0:
                print(f"前5个元素: {value[:5]}")
        
        # 如果是字典
        elif isinstance(value, dict):
            print(f"类型: dict")
            print(f"keys: {list(value.keys())}")
        
        # 其他类型（数字、字符串等）
        else:
            print(f"类型: {type(value).__name__}")
            print(f"值: {value}")
    
    print("\n" + "=" * 60)
    print("检查完成！")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python show_pkl_full.py <pkl文件路径>")
        print("\n示例:")
        print("  python show_pkl_full.py 36_01.pkl")
        print("  python show_pkl_full.py 114_08.pkl")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    show_pkl_full(pkl_path)