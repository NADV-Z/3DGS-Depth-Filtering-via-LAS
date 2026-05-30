import subprocess

# output保存路径
model_path = r'F:\3dgs\new_try\gaussian-splatting-main\output\80a4a348-a\point_cloud\iteration_10000'

# 脚本执行
command = f'SIBR_gaussianViewer_app.exe -m {model_path}'
run_path = 'external/viewers/bin'
subprocess.run(command, shell=True, cwd=run_path)
