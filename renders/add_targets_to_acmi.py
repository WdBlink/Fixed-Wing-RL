#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACMI目标位置可视化工具

该脚本用于将tracking任务中的目标位置信息添加到ACMI文件中，
使得在TacView中可以同时可视化飞机轨迹和目标位置。

作者: wdblink
日期: 2025
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional


class ACMITargetVisualizer:
    """ACMI目标位置可视化器
    
    该类负责读取ACMI文件和目标位置数据，将目标位置作为新的对象
    添加到ACMI文件中，生成包含目标位置可视化的新ACMI文件。
    """
    
    def __init__(self, result_dir: str = "./result", tracks_dir: str = "./tracks"):
        """初始化可视化器
        
        Args:
            result_dir: 包含.npy结果文件的目录路径
            tracks_dir: 包含ACMI文件的目录路径
        """
        self.result_dir = Path(result_dir).resolve()
        self.tracks_dir = Path(tracks_dir).resolve()
        
        # 验证目录存在
        if not self.result_dir.exists():
            raise FileNotFoundError(f"结果目录不存在: {self.result_dir}")
        if not self.tracks_dir.exists():
            raise FileNotFoundError(f"轨迹目录不存在: {self.tracks_dir}")
    
    def load_target_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """加载目标位置数据
        
        Returns:
            包含目标北向位置、东向位置和高度的元组，如果文件不存在则返回None
        """
        target_npos = self._load_if_exists("target_npos.npy")
        target_epos = self._load_if_exists("target_epos.npy")
        target_altitude = self._load_if_exists("target_altitude.npy")
        
        return target_npos, target_epos, target_altitude
    
    def _load_if_exists(self, filename: str) -> Optional[np.ndarray]:
        """如果文件存在则加载numpy数组
        
        Args:
            filename: 文件名
            
        Returns:
            加载的numpy数组，如果文件不存在则返回None
        """
        file_path = self.result_dir / filename
        if file_path.exists():
            return np.load(file_path)
        else:
            print(f"警告: {filename} 不存在，跳过加载")
            return None
    
    def convert_coordinates(self, npos: np.ndarray, epos: np.ndarray, altitude: np.ndarray) -> List[Tuple[float, float, float]]:
        """将NED坐标转换为ACMI格式的经纬度坐标
        
        Args:
            npos: 北向位置数组 (feet)
            epos: 东向位置数组 (feet)
            altitude: 高度数组 (feet)
            
        Returns:
            包含(经度, 纬度, 高度)的坐标列表
        """
        # 参考点 (假设的起始位置)
        ref_lat = 0.0  # 参考纬度
        ref_lon = 0.0  # 参考经度
        
        # 将feet转换为度 (粗略转换)
        # 1度纬度 ≈ 364,000 feet
        # 1度经度 ≈ 364,000 * cos(lat) feet
        feet_per_degree_lat = 364000.0
        feet_per_degree_lon = 364000.0  # 简化处理，假设在赤道附近
        
        coordinates = []
        for i in range(len(npos)):
            lat = ref_lat + npos[i] / feet_per_degree_lat
            lon = ref_lon + epos[i] / feet_per_degree_lon
            alt = altitude[i]  # 保持feet单位
            coordinates.append((lon, lat, alt))
        
        return coordinates
    
    def add_targets_to_acmi(self, acmi_file: Path, target_coords: List[Tuple[float, float, float]], output_file: Path):
        """将目标位置添加到ACMI文件中
        
        Args:
            acmi_file: 原始ACMI文件路径
            target_coords: 目标坐标列表
            output_file: 输出ACMI文件路径
        """
        with open(acmi_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 找到时间步长
        time_step = 0.02  # 默认时间步长
        for line in lines:
            if line.startswith('#'):
                try:
                    time_step = float(line[1:].strip())
                    break
                except ValueError:
                    continue
        
        # 准备输出内容
        output_lines = []
        time_index = 0
        
        for line in lines:
            output_lines.append(line)
            
            # 如果是时间标记行，添加目标位置
            if line.startswith('#') and time_index < len(target_coords):
                lon, lat, alt = target_coords[time_index]
                # 添加目标对象 (使用ID 200)
                target_line = f"200,T={lon}|{lat}|{alt}|0.0|0.0|0.0,Name=Target,Color=Blue,Type=Ground+Static\n"
                output_lines.append(target_line)
                time_index += 1
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
        
        print(f"已生成包含目标位置的ACMI文件: {output_file}")
    
    def process_all_acmi_files(self):
        """处理所有ACMI文件，添加目标位置信息"""
        # 加载目标数据
        target_npos, target_epos, target_altitude = self.load_target_data()
        
        if target_npos is None or target_epos is None or target_altitude is None:
            print("错误: 缺少必要的目标位置数据文件")
            return
        
        # 转换坐标
        target_coords = self.convert_coordinates(target_npos, target_epos, target_altitude)
        
        # 处理所有ACMI文件
        acmi_files = list(self.tracks_dir.glob("*.acmi"))
        if not acmi_files:
            print("警告: 未找到ACMI文件")
            return
        
        # 创建输出目录
        output_dir = self.tracks_dir / "with_targets"
        output_dir.mkdir(exist_ok=True)
        
        for acmi_file in acmi_files:
            output_file = output_dir / f"target_{acmi_file.name}"
            self.add_targets_to_acmi(acmi_file, target_coords, output_file)
        
        print(f"\n处理完成！生成了 {len(acmi_files)} 个包含目标位置的ACMI文件")
        print(f"输出目录: {output_dir}")
        print("\n使用方法:")
        print("1. 打开TacView软件")
        print("2. 加载生成的target_*.acmi文件")
        print("3. 红色对象为F16飞机，蓝色对象为目标位置")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="将目标位置添加到ACMI文件中进行可视化")
    parser.add_argument("--result-dir", type=str, default="./result",
                        help="包含.npy结果文件的目录 (默认: ./result)")
    parser.add_argument("--tracks-dir", type=str, default="./tracks",
                        help="包含ACMI文件的目录 (默认: ./tracks)")
    
    args = parser.parse_args()
    
    try:
        visualizer = ACMITargetVisualizer(args.result_dir, args.tracks_dir)
        visualizer.process_all_acmi_files()
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()