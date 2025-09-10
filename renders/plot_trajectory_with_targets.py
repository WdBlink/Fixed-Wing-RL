#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞行轨迹与目标位置2D可视化工具

该脚本用于生成包含飞机轨迹和目标位置的2D可视化图表，
帮助分析tracking任务中agent的跟踪性能。

作者: wdblink
日期: 2025
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import matplotlib.patches as patches


class TrajectoryVisualizer:
    """飞行轨迹可视化器
    
    该类负责读取飞行数据和目标位置数据，生成2D轨迹可视化图表，
    包括位置轨迹、高度变化、速度变化等多种视图。
    """
    
    def __init__(self, result_dir: str = "./result"):
        """初始化可视化器
        
        Args:
            result_dir: 包含.npy结果文件的目录路径
        """
        self.result_dir = Path(result_dir).resolve()
        
        if not self.result_dir.exists():
            raise FileNotFoundError(f"结果目录不存在: {self.result_dir}")
    
    def load_data(self) -> dict:
        """加载所有相关数据
        
        Returns:
            包含所有加载数据的字典
        """
        data = {}
        
        # 飞机位置数据
        data['npos'] = self._load_if_exists("npos.npy")
        data['epos'] = self._load_if_exists("epos.npy")
        data['altitude'] = self._load_if_exists("altitude.npy")
        
        # 目标位置数据
        data['target_npos'] = self._load_if_exists("target_npos.npy")
        data['target_epos'] = self._load_if_exists("target_epos.npy")
        data['target_altitude'] = self._load_if_exists("target_altitude.npy")
        
        # 其他飞行参数
        data['vt'] = self._load_if_exists("vt.npy")
        data['target_vt'] = self._load_if_exists("target_vt.npy")
        data['yaw'] = self._load_if_exists("yaw.npy")
        data['target_heading'] = self._load_if_exists("target_heading.npy")
        
        return data
    
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
            print(f"警告: {filename} 不存在")
            return None
    
    def plot_2d_trajectory(self, data: dict, save_path: Optional[str] = None):
        """绘制2D轨迹图
        
        Args:
            data: 包含飞行数据的字典
            save_path: 保存图片的路径，如果为None则显示图片
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('飞行轨迹与目标位置分析', fontsize=16, fontweight='bold')
        
        # 1. 水平轨迹图 (North vs East)
        if data['npos'] is not None and data['epos'] is not None:
            ax1.plot(data['epos'], data['npos'], 'r-', linewidth=2, label='飞机轨迹', alpha=0.8)
            ax1.scatter(data['epos'][0], data['npos'][0], color='green', s=100, marker='o', label='起始点', zorder=5)
            ax1.scatter(data['epos'][-1], data['npos'][-1], color='red', s=100, marker='s', label='结束点', zorder=5)
            
            # 添加目标位置
            if data['target_npos'] is not None and data['target_epos'] is not None:
                # 只显示与实际轨迹长度匹配的目标点
                actual_len = len(data['npos'])
                target_len = min(len(data['target_npos']), actual_len)
                
                ax1.plot(data['target_epos'][:target_len], data['target_npos'][:target_len], 
                        'b--', linewidth=2, label='目标轨迹', alpha=0.7)
                ax1.scatter(data['target_epos'][0], data['target_npos'][0], 
                           color='blue', s=80, marker='^', label='目标起始', zorder=5)
                
                # 计算跟踪误差
                pos_error = np.sqrt((data['epos'][:target_len] - data['target_epos'][:target_len])**2 + 
                                   (data['npos'][:target_len] - data['target_npos'][:target_len])**2)
                avg_error = np.mean(pos_error)
                ax1.text(0.02, 0.98, f'平均位置误差: {avg_error:.1f} feet', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax1.set_xlabel('东向位置 (feet)')
            ax1.set_ylabel('北向位置 (feet)')
            ax1.set_title('水平轨迹图')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.axis('equal')
        
        # 2. 高度变化图
        if data['altitude'] is not None:
            time_steps = np.arange(len(data['altitude'])) * 0.02  # 假设时间步长为0.02s
            ax2.plot(time_steps, data['altitude'], 'r-', linewidth=2, label='实际高度')
            
            if data['target_altitude'] is not None:
                actual_len = len(data['altitude'])
                target_len = min(len(data['target_altitude']), actual_len)
                ax2.plot(time_steps[:target_len], data['target_altitude'][:target_len], 
                        'b--', linewidth=2, label='目标高度')
                
                # 计算高度误差
                alt_error = np.abs(data['altitude'][:target_len] - data['target_altitude'][:target_len])
                avg_alt_error = np.mean(alt_error)
                ax2.text(0.02, 0.98, f'平均高度误差: {avg_alt_error:.1f} feet', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax2.set_xlabel('时间 (s)')
            ax2.set_ylabel('高度 (feet)')
            ax2.set_title('高度变化')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 3. 速度变化图
        if data['vt'] is not None:
            time_steps = np.arange(len(data['vt'])) * 0.02
            ax3.plot(time_steps, data['vt'], 'r-', linewidth=2, label='实际速度')
            
            if data['target_vt'] is not None:
                actual_len = len(data['vt'])
                target_len = min(len(data['target_vt']), actual_len)
                ax3.plot(time_steps[:target_len], data['target_vt'][:target_len], 
                        'b--', linewidth=2, label='目标速度')
                
                # 计算速度误差
                vt_error = np.abs(data['vt'][:target_len] - data['target_vt'][:target_len])
                avg_vt_error = np.mean(vt_error)
                ax3.text(0.02, 0.98, f'平均速度误差: {avg_vt_error:.1f} feet/s', 
                        transform=ax3.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            ax3.set_xlabel('时间 (s)')
            ax3.set_ylabel('速度 (feet/s)')
            ax3.set_title('速度变化')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 4. 航向变化图
        if data['yaw'] is not None:
            time_steps = np.arange(len(data['yaw'])) * 0.02
            ax4.plot(time_steps, data['yaw'] * 180 / np.pi, 'r-', linewidth=2, label='实际航向')
            
            if data['target_heading'] is not None:
                actual_len = len(data['yaw'])
                target_len = min(len(data['target_heading']), actual_len)
                ax4.plot(time_steps[:target_len], data['target_heading'][:target_len] * 180 / np.pi, 
                        'b--', linewidth=2, label='目标航向')
                
                # 计算航向误差
                heading_error = np.abs(data['yaw'][:target_len] - data['target_heading'][:target_len]) * 180 / np.pi
                avg_heading_error = np.mean(heading_error)
                ax4.text(0.02, 0.98, f'平均航向误差: {avg_heading_error:.1f}°', 
                        transform=ax4.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            ax4.set_xlabel('时间 (s)')
            ax4.set_ylabel('航向 (度)')
            ax4.set_title('航向变化')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"轨迹图已保存到: {save_path}")
        else:
            plt.show()
    
    def plot_3d_trajectory(self, data: dict, save_path: Optional[str] = None):
        """绘制3D轨迹图
        
        Args:
            data: 包含飞行数据的字典
            save_path: 保存图片的路径，如果为None则显示图片
        """
        if data['npos'] is None or data['epos'] is None or data['altitude'] is None:
            print("缺少3D轨迹所需的位置数据")
            return
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制飞机轨迹
        ax.plot(data['epos'], data['npos'], data['altitude'], 
                'r-', linewidth=3, label='飞机轨迹', alpha=0.8)
        
        # 标记起始和结束点
        ax.scatter(data['epos'][0], data['npos'][0], data['altitude'][0], 
                  color='green', s=100, label='起始点')
        ax.scatter(data['epos'][-1], data['npos'][-1], data['altitude'][-1], 
                  color='red', s=100, label='结束点')
        
        # 绘制目标轨迹
        if (data['target_npos'] is not None and 
            data['target_epos'] is not None and 
            data['target_altitude'] is not None):
            
            actual_len = len(data['npos'])
            target_len = min(len(data['target_npos']), actual_len)
            
            ax.plot(data['target_epos'][:target_len], 
                   data['target_npos'][:target_len], 
                   data['target_altitude'][:target_len], 
                   'b--', linewidth=2, label='目标轨迹', alpha=0.7)
            
            ax.scatter(data['target_epos'][0], data['target_npos'][0], data['target_altitude'][0], 
                      color='blue', s=80, marker='^', label='目标起始')
        
        ax.set_xlabel('东向位置 (feet)')
        ax.set_ylabel('北向位置 (feet)')
        ax.set_zlabel('高度 (feet)')
        ax.set_title('3D飞行轨迹', fontsize=14, fontweight='bold')
        ax.legend()
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D轨迹图已保存到: {save_path}")
        else:
            plt.show()
    
    def generate_summary_report(self, data: dict) -> str:
        """生成跟踪性能摘要报告
        
        Args:
            data: 包含飞行数据的字典
            
        Returns:
            格式化的摘要报告字符串
        """
        report = ["\n=== 跟踪任务性能摘要报告 ==="]
        
        if (data['npos'] is not None and data['epos'] is not None and 
            data['target_npos'] is not None and data['target_epos'] is not None):
            
            actual_len = len(data['npos'])
            target_len = min(len(data['target_npos']), actual_len)
            
            # 位置跟踪误差
            pos_error = np.sqrt((data['epos'][:target_len] - data['target_epos'][:target_len])**2 + 
                               (data['npos'][:target_len] - data['target_npos'][:target_len])**2)
            report.append(f"平均位置误差: {np.mean(pos_error):.2f} feet")
            report.append(f"最大位置误差: {np.max(pos_error):.2f} feet")
            report.append(f"最小位置误差: {np.min(pos_error):.2f} feet")
            
            # 高度跟踪误差
            if data['altitude'] is not None and data['target_altitude'] is not None:
                alt_error = np.abs(data['altitude'][:target_len] - data['target_altitude'][:target_len])
                report.append(f"平均高度误差: {np.mean(alt_error):.2f} feet")
                report.append(f"最大高度误差: {np.max(alt_error):.2f} feet")
            
            # 速度跟踪误差
            if data['vt'] is not None and data['target_vt'] is not None:
                vt_error = np.abs(data['vt'][:target_len] - data['target_vt'][:target_len])
                report.append(f"平均速度误差: {np.mean(vt_error):.2f} feet/s")
                report.append(f"最大速度误差: {np.max(vt_error):.2f} feet/s")
            
            # 航向跟踪误差
            if data['yaw'] is not None and data['target_heading'] is not None:
                heading_error = np.abs(data['yaw'][:target_len] - data['target_heading'][:target_len]) * 180 / np.pi
                report.append(f"平均航向误差: {np.mean(heading_error):.2f}°")
                report.append(f"最大航向误差: {np.max(heading_error):.2f}°")
            
            # 任务完成情况
            final_pos_error = pos_error[-1]
            if final_pos_error < 1000:  # 1000 feet以内认为成功
                report.append(f"\n任务状态: 成功 (最终误差: {final_pos_error:.2f} feet)")
            else:
                report.append(f"\n任务状态: 未完成 (最终误差: {final_pos_error:.2f} feet)")
        
        report.append("=" * 40)
        return "\n".join(report)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成飞行轨迹与目标位置的可视化图表")
    parser.add_argument("--result-dir", type=str, default="./result",
                        help="包含.npy结果文件的目录 (默认: ./result)")
    parser.add_argument("--output-dir", type=str, default="./plots",
                        help="输出图片的目录 (默认: ./plots)")
    parser.add_argument("--show", action="store_true",
                        help="显示图片而不是保存")
    
    args = parser.parse_args()
    
    try:
        visualizer = TrajectoryVisualizer(args.result_dir)
        data = visualizer.load_data()
        
        # 创建输出目录
        if not args.show:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # 生成2D轨迹图
        save_path_2d = None if args.show else str(output_dir / "trajectory_2d.png")
        visualizer.plot_2d_trajectory(data, save_path_2d)
        
        # 生成3D轨迹图
        save_path_3d = None if args.show else str(output_dir / "trajectory_3d.png")
        visualizer.plot_3d_trajectory(data, save_path_3d)
        
        # 生成摘要报告
        report = visualizer.generate_summary_report(data)
        print(report)
        
        if not args.show:
            # 保存报告到文件
            with open(output_dir / "tracking_report.txt", 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n报告已保存到: {output_dir / 'tracking_report.txt'}")
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()