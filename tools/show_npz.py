import numpy as np

def show_npz(npz_path):
    """
    展示npz文件的内容结构
    """
    # 读取npz文件
    data = np.load(npz_path, allow_pickle=True)
    
    print("=== npz文件数据结构 ===")
    print(f"keys: {list(data.keys())}")
    
    # 遍历所有数组
    for key in data.keys():
        arr = data[key]
        print(f"\n--- {key} ---")
        print(f"shape: {arr.shape}")
        print(f"dtype: {arr.dtype}")
        
        # 显示前几帧数据（如果是二维数组）
        if arr.ndim >= 1:
            if arr.ndim == 1:
                print(f"前5个元素:\n{arr[:5]}")
            elif arr.ndim >= 2:
                print(f"前3帧数据:\n{arr[:3]}")
                
                # 如果是四元数数据，检查模长
                if arr.shape[-1] == 4 and key.lower().find('rot') >= 0:
                    print(f"第一帧四元数模长: {np.linalg.norm(arr[0]):.6f}")
        
        # 显示统计信息
        if np.issubdtype(arr.dtype, np.number):
            print(f"min: {arr.min():.6f}")
            print(f"max: {arr.max():.6f}")
            print(f"mean: {arr.mean():.6f}")
            print(f"std: {arr.std():.6f}")
    
    data.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python show_npz.py <npz文件路径>")
        print("\n示例:")
        print("  python show_npz.py data/output.npz")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    show_npz(npz_path)