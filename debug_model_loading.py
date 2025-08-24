#!/usr/bin/env python3
"""
调试模型加载问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.agriculture.era5_soil_moisture_predictor import ERA5SoilMoisturePredictor
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_model_loading():
    """调试模型加载"""
    try:
        logger.info("🔍 开始调试模型加载...")
        
        # 创建预测器
        predictor = ERA5SoilMoisturePredictor()
        
        # 检查模型文件
        model_files = ['current_soil_moisture_model.pth', 'best_model.pth']
        
        for model_file in model_files:
            logger.info(f"📁 检查模型文件: {model_file}")
            
            if os.path.exists(model_file):
                logger.info(f"✅ 文件存在: {model_file}")
                
                # 尝试加载
                try:
                    predictor.load_model(model_file)
                    logger.info(f"✅ 成功加载模型: {model_file}")
                    return True
                except Exception as e:
                    logger.error(f"❌ 加载失败: {e}")
                    logger.error(f"详细错误: {type(e).__name__}: {str(e)}")
            else:
                logger.warning(f"⚠️ 文件不存在: {model_file}")
        
        # 检查模型目录
        logger.info("📁 检查模型目录...")
        model_dir = "models/era5_soil_moisture"
        if os.path.exists(model_dir):
            logger.info(f"✅ 模型目录存在: {model_dir}")
            files = os.listdir(model_dir)
            logger.info(f"📋 目录内容: {files}")
            
            # 尝试加载best_model.pth
            best_model_path = os.path.join(model_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                logger.info(f"✅ 找到best_model.pth: {best_model_path}")
                try:
                    predictor.load_model("best_model.pth")
                    logger.info("✅ 成功加载best_model.pth")
                    return True
                except Exception as e:
                    logger.error(f"❌ 加载best_model.pth失败: {e}")
                    logger.error(f"详细错误: {type(e).__name__}: {str(e)}")
        else:
            logger.error(f"❌ 模型目录不存在: {model_dir}")
        
        return False
        
    except Exception as e:
        logger.error(f"❌ 调试失败: {e}")
        return False

if __name__ == "__main__":
    debug_model_loading()
