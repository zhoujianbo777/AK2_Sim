@echo off
chcp 65001 >nul
echo ============================================================
echo  AK2 AI模型训练环境配置脚本
echo  文档依据: AK2 AI模型开发说明.md V1.1
echo  GPU: NVIDIA 3060 Ti / CUDA 12.1
echo  目标环境: conda ak2_sim (Python 3.10)
echo ============================================================
echo.

REM 激活目标环境
call conda activate ak2_sim
if %errorlevel% neq 0 (
    echo [ERROR] 无法激活 ak2_sim 环境
    pause
    exit /b 1
)

echo [1/5] 卸载 CPU 版 PyTorch...
pip uninstall torch torchvision -y 2>nul

echo.
echo [2/5] 安装 CUDA 12.1 版 PyTorch 2.x...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3/5] 安装模型训练核心依赖...
pip install scikit-learn>=1.3.0 onnx>=1.16.0 pandas>=2.1.0 seaborn>=0.13.0

echo.
echo [4/5] 安装实验管理工具 MLflow...
pip install mlflow>=2.14.0

echo.
echo [5/5] 升级关键包到最新兼容版本...
pip install --upgrade numpy scipy matplotlib tqdm h5py pyyaml

echo.
echo ============================================================
echo  环境配置完成！执行验证...
echo ============================================================
echo.

python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'GPU 名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ============================================================
echo  验证完成！
echo.
echo  如需手动验证，请运行:
echo    conda activate ak2_sim
echo    python -c "import torch; print(torch.cuda.is_available())"
echo ============================================================

pause
