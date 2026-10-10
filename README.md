# Crazyflie 2.1+ RL Hover

当前保留版本：**d4620ad5**。适用于有刷 Crazyflie 2.1+、32.4 g、Lighthouse deck v4。

起飞和悬停使用同一 RL 四电机策略。用户已报告能够稳定起飞；偏航控制和自主降落仍未完成。该版本属于实验固件。

[结果说明](RESULTS.md) · [完整运行包](https://github.com/Meng-Zhao-flie/RL_takeoff/releases/tag/hover-d4620ad5)

## 准备

macOS 和 Windows Anaconda Prompt / PowerShell 使用相同命令。首次建立环境：

```text
conda create -n crazyflie python=3.11 -y
conda activate crazyflie
python -m pip install -r requirements-flight.txt
python setup_release.py
```

已有环境只需激活并安装依赖。准备脚本自动下载、校验并解压保留版本，不连接飞机。

离线检查：

```text
python run_hover.py
```

## 刷写

关闭 cfclient 和其他无线客户端，只保留一个 Crazyradio。

冷启动：先关机，执行命令；出现倒计时后长按电源键约 3 秒，M2 蓝灯闪烁后松开。

```text
python -m cfloader flash firmware/simple_hover_trial/build/firmware.zip -c
```

热启动：飞机正常开机后执行：

```text
python -m cfloader flash firmware/simple_hover_trial/build/firmware.zip -w radio://0/80/2M/E7E7E7E701
```

完整包包含官方 2026.08 nRF51 和 Lighthouse V7。冷刷会跳过 deck 更新；需要更新 deck 时使用热刷。

## 飞行

刷写完成后正常重启飞机，再单独执行：

```text
conda activate crazyflie
python run_hover.py --execute
```

默认 URI 为 `radio://0/80/2M/E7E7E7E701`。其他飞机请修改 `deployment/current/RL_takeoff/flight_config.json`。

**Ctrl+C 会停电机，不执行自主降落。** 当前版本没有训练降落。Lighthouse 质量和电压保留为诊断信息。

源代码、权重、构建证明和仿真证据随完整运行包保存；SHA 见 [CURRENT_HOVER_RELEASE.json](CURRENT_HOVER_RELEASE.json)。
