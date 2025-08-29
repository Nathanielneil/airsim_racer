# AirSim RACER

基于AirSim-UE仿真平台的RACER多无人机协同探索系统的Windows部署版本。

## 项目概述

将原始的基于ROS的RACER系统重构为基于AirSim的Python实现，适用于Windows平台部署和AirSim-UE仿真环境。

## 环境要求

- Windows 10/11
- Python 3.9
- AirSim-UE仿真器
- Conda包管理器

## 安装指南

### 1. 创建Conda环境

```bash
conda env create -f environment.yml
conda activate airsim_racer
```

或使用pip安装：

```bash
pip install -r requirements.txt
```

### 2. AirSim配置

确保AirSim设置文件配置正确。参考 `config/airsim_settings.json`。

### 3. 运行系统

```bash
python main.py
```

## 项目结构

- `src/` - 核心代码模块
  - `exploration/` - 探索管理器
  - `planning/` - 路径规划算法
  - `perception/` - 感知模块
  - `airsim_interface/` - AirSim接口
- `config/` - 配置文件
- `scripts/` - 脚本工具
- `tests/` - 测试代码

## 项目信息

本项目基于SYSU-STAR的RACER项目重构：
- 原项目地址：https://github.com/SYSU-STAR/RACER
- 论文：RACER: Rapid Collaborative Exploration with a Decentralized Multi-UAV System (IEEE T-RO 2023)
