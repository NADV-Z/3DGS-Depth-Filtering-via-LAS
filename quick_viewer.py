import subprocess
import sys
import os
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SIBR Gaussian Viewer')
    parser.add_argument('-m', '--model', required=True, help='模型输出目录路径')
    args = parser.parse_args()
    
    # 获取当前脚本的绝对路径
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"脚本目录: {current_script_dir}")
    
    # SIBR viewer可执行文件路径
    sibr_exe_paths = [
        os.path.join(current_script_dir, "viewers", "bin", "SIBR_gaussianViewer_app.exe"),
        os.path.join(current_script_dir, "viewers", "bin", "SIBR_gaussianViewer_app"),
        os.path.join(current_script_dir, "SIBR_viewers", "install", "bin", "SIBR_gaussianViewer_app.exe"),
        os.path.join(current_script_dir, "SIBR_viewers", "install", "bin", "SIBR_gaussianViewer_app"),
    ]
    
    sibr_exe = None
    for path in sibr_exe_paths:
        if os.path.exists(path):
            sibr_exe = path
            print(f"找到SIBR viewer: {sibr_exe}")
            break
    
    if sibr_exe is None:
        print("❌ 找不到SIBR viewer可执行文件")
        print("搜索过的路径:")
        for path in sibr_exe_paths:
            print(f"  - {path}")
        print("\n请检查SIBR viewer是否正确安装")
        return 1
    
    # 检查模型目录
    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        print(f"❌ 模型目录不存在: {model_path}")
        return 1
    
    print(f"模型目录: {model_path}")
    
    # 检查必要的文件
    ply_file = os.path.join(model_path, "point_cloud.ply")
    if not os.path.exists(ply_file):
        print(f"❌ 找不到点云文件: {ply_file}")
        return 1
    
    cfg_file = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_file):
        print("⚠️  cfg_args文件不存在，正在创建...")
        create_cfg_args(cfg_file)
    
    # 构建命令
    # 使用列表形式而不是字符串，避免shell解析问题
    cmd_args = [
        sibr_exe,
        "-m", model_path
    ]
    
    print(f"执行命令: {' '.join(cmd_args)}")
    print(f"工作目录: {current_script_dir}")
    
    try:
        # 方法1: 直接运行，不使用shell
        print("尝试方法1: 直接运行...")
        result = subprocess.run(
            cmd_args,
            cwd=current_script_dir,
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"方法1失败，返回码: {result.returncode}")
            
            # 方法2: 使用shell，完整引号包围路径
            print("尝试方法2: 使用shell...")
            shell_cmd = f'"{sibr_exe}" -m "{model_path}"'
            result = subprocess.run(
                shell_cmd,
                shell=True,
                cwd=current_script_dir,
                capture_output=False,
                text=True
            )
            
            if result.returncode != 0:
                print(f"方法2也失败，返回码: {result.returncode}")
                return result.returncode
    
    except FileNotFoundError as e:
        print(f"❌ 文件未找到错误: {e}")
        return 1
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        return 1
    
    print("✅ SIBR viewer启动成功")
    return 0

def create_cfg_args(cfg_file):
    """创建基本的cfg_args配置文件"""
    cfg_content = """Namespace(data_device='cuda', debug=False, debug_from=-1, detect_anomaly=False, eval=False, images='images', lod=0, model_path='', quiet=False, render_process=False, resolution=-1, sh_degree=3, source_path='', test_iterations=[7000, 30000], white_background=False)"""
    
    try:
        with open(cfg_file, 'w', encoding='utf-8') as f:
            f.write(cfg_content)
        print(f"✅ 已创建配置文件: {cfg_file}")
    except Exception as e:
        print(f"❌ 创建配置文件失败: {e}")

if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        print("\n建议:")
        print("1. 检查CUDA版本是否兼容 (需要CUDA 12.x)")
        print("2. 尝试重新编译SIBR viewer")
        print("3. 使用Python替代可视化方案")
        input("按回车键退出...")
    sys.exit(exit_code)