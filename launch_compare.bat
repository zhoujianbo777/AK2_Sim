@echo off
REM AK2 仿真软件 双实例对比模式启动脚本
REM 先后启动传统算法实例（进程A）和AI算法实例（进程B）

title AK2 Compare Launcher

echo ==========================================
echo   AK2 超声波感知系统 双实例对比模式
echo ==========================================
echo.

REM 检查Python环境
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 python 命令，请激活 conda 环境后重试
    echo 提示: conda activate ak2_sim
    pause
    exit /b 1
)

REM 检查主程序是否存在
if not exist "sim_main.py" (
    echo [错误] 未找到 sim_main.py，请在软件根目录下运行此脚本
    pause
    exit /b 1
)

echo [1/2] 启动传统算法实例（进程A - Master）...
start "AK2-Traditional" python sim_main.py --mode traditional --ipc-role master

REM 延迟1.5秒，等待进程A建立命名管道
timeout /t 2 /nobreak >nul

echo [2/2] 启动AI算法实例（进程B - Slave）...
start "AK2-AI" python sim_main.py --mode ai --ipc-role slave

echo.
echo 两个实例已启动，请在各自窗口中加载相同数据集后点击播放。
echo 主控制面板（W1）以传统算法窗口为准，可同步控制双侧回放。
echo.
pause
