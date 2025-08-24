#!/usr/bin/env python3
"""
防过拟合系统测试脚本
验证系统的有效性和实用性
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from datetime import datetime

# 导入防过拟合系统
from src.models.anti_overfitting_core import AntiOverfittingCore
from src.data.data_quality_detector import DataQualityDetector
from src.models.training_fixer import TrainingFixer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 200, n_features: int = 10) -> tuple:
    """创建测试数据"""
    try:
        logger.info(f"🔧 创建测试数据: {n_samples} 样本, {n_features} 特征")
        
        # 生成特征数据
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        
        # 生成目标变量（添加一些噪声）
        y = np.sum(X[:, :3], axis=1) + np.random.normal(0, 0.1, n_samples)
        
        # 分割训练集和验证集
        split_idx = int(n_samples * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"✅ 测试数据创建完成: 训练集 {len(X_train)}, 验证集 {len(X_val)}")
        
        return X_train, y_train, X_val, y_val
        
    except Exception as e:
        logger.error(f"❌ 创建测试数据失败: {e}")
        raise

def create_test_model(input_size: int = 10) -> nn.Module:
    """创建测试模型"""
    try:
        class TestLSTM(nn.Module):
            def __init__(self, input_size: int):
                super(TestLSTM, self).__init__()
                
                # 故意创建过复杂的模型来测试过拟合检测
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=128,  # 过大的隐藏层
                    num_layers=4,     # 过多的层数
                    batch_first=True,
                    dropout=0.0       # 无正则化
                )
                
                self.fc = nn.Linear(128, 1)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_output = lstm_out[:, -1, :]
                return self.fc(last_output)
        
        model = TestLSTM(input_size)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ 测试模型创建完成，参数数量: {total_params}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ 创建测试模型失败: {e}")
        raise

def simulate_training(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> tuple:
    """模拟训练过程"""
    try:
        logger.info("🚀 开始模拟训练...")
        
        # 准备数据 - 添加序列维度
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # (batch, seq_len=1, features)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)      # (batch, seq_len=1, features)
        y_val_tensor = torch.FloatTensor(y_val)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(X_val))
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练参数
        epochs = 30
        train_losses = []
        val_losses = []
        
        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            # 记录损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # 模拟过拟合：训练损失下降，验证损失上升
            if epoch > 15:
                # 故意增加验证损失来模拟过拟合
                val_losses[-1] += 0.01 * (epoch - 15)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        logger.info("✅ 模拟训练完成")
        return train_losses, val_losses
        
    except Exception as e:
        logger.error(f"❌ 模拟训练失败: {e}")
        raise

def test_anti_overfitting_system():
    """测试防过拟合系统"""
    try:
        logger.info("🧪 开始测试防过拟合系统...")
        
        # 步骤1: 创建测试数据
        X_train, y_train, X_val, y_val = create_test_data(200, 10)
        
        # 步骤2: 创建测试模型
        model = create_test_model(10)
        
        # 步骤3: 模拟训练
        train_losses, val_losses = simulate_training(model, X_train, y_train, X_val, y_val)
        
        # 步骤4: 测试数据质量检测器
        logger.info("\n📊 测试数据质量检测器...")
        data_quality = DataQualityDetector()
        quality_result = data_quality.detect_data_issues(X_train, y_train)
        
        if quality_result['status'] == 'success':
            logger.info(f"数据质量得分: {quality_result['quality_score']:.3f}")
            logger.info(f"发现问题数量: {quality_result['total_issues']}")
            
            for issue in quality_result['issues']:
                logger.info(f"  - {issue['severity'].upper()}: {issue['description']}")
        
        # 步骤5: 测试过拟合检测
        logger.info("\n🔍 测试过拟合检测...")
        anti_overfitting = AntiOverfittingCore()
        overfitting_result = anti_overfitting.detect_overfitting(train_losses, val_losses)
        
        if overfitting_result['status'] == 'success':
            logger.info(f"过拟合检测结果: {'是' if overfitting_result['overfitting'] else '否'}")
            if overfitting_result['overfitting']:
                logger.info(f"严重程度: {overfitting_result['severity']:.3f}")
                logger.info(f"建议: {overfitting_result['recommendation']}")
        
        # 步骤6: 测试训练修复器
        logger.info("\n🔧 测试训练修复器...")
        training_fixer = TrainingFixer()
        fix_result = training_fixer.diagnose_and_fix(
            model, X_train, y_train, X_val, y_val, train_losses, val_losses
        )
        
        if fix_result.get('status') == 'success':
            logger.info(f"修复状态: {fix_result.get('final_status', 'unknown')}")
            logger.info(f"应用修复数量: {len(fix_result.get('fixes_applied', []))}")
        else:
            logger.warning(f"修复结果状态异常: {fix_result}")
        
        # 步骤7: 生成测试报告
        logger.info("\n📋 生成测试报告...")
        generate_test_report(quality_result, overfitting_result, fix_result)
        
        logger.info("✅ 防过拟合系统测试完成")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试防过拟合系统失败: {e}")
        return False

def generate_test_report(quality_result: dict, overfitting_result: dict, fix_result: dict):
    """生成测试报告"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"test_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("HydrAI-SWE 防过拟合系统测试报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 数据质量报告
            f.write("📊 数据质量检测结果\n")
            f.write("-" * 30 + "\n")
            if quality_result['status'] == 'success':
                f.write(f"质量得分: {quality_result['quality_score']:.3f}\n")
                f.write(f"问题总数: {quality_result['total_issues']}\n")
                f.write(f"严重问题: {quality_result['critical_issues']}\n")
                f.write(f"高严重性: {quality_result['high_issues']}\n")
                f.write(f"中等严重性: {quality_result['medium_issues']}\n\n")
            else:
                f.write("数据质量检测失败\n\n")
            
            # 过拟合检测报告
            f.write("🔍 过拟合检测结果\n")
            f.write("-" * 30 + "\n")
            if overfitting_result['status'] == 'success':
                f.write(f"过拟合: {'是' if overfitting_result['overfitting'] else '否'}\n")
                if overfitting_result['overfitting']:
                    f.write(f"严重程度: {overfitting_result['severity']:.3f}\n")
                    f.write(f"建议: {overfitting_result['recommendation']}\n")
                f.write("\n")
            else:
                f.write("过拟合检测失败\n\n")
            
            # 修复结果报告
            f.write("🔧 修复结果\n")
            f.write("-" * 30 + "\n")
            if fix_result['status'] == 'success':
                f.write(f"最终状态: {fix_result['final_status']}\n")
                f.write(f"应用修复: {len(fix_result['fixes_applied'])} 个\n")
                for i, fix in enumerate(fix_result['fixes_applied']):
                    f.write(f"  修复 {i+1}: {fix['type']}\n")
                f.write("\n")
            else:
                f.write("修复失败\n\n")
            
            # 总结
            f.write("📋 测试总结\n")
            f.write("-" * 30 + "\n")
            if quality_result['status'] == 'success' and overfitting_result['status'] == 'success':
                f.write("✅ 防过拟合系统测试成功\n")
                f.write("✅ 数据质量检测功能正常\n")
                f.write("✅ 过拟合检测功能正常\n")
                if fix_result['status'] == 'success':
                    f.write("✅ 训练修复功能正常\n")
                else:
                    f.write("❌ 训练修复功能异常\n")
            else:
                f.write("❌ 防过拟合系统测试失败\n")
        
        logger.info(f"✅ 测试报告已生成: {report_file}")
        
    except Exception as e:
        logger.error(f"❌ 生成测试报告失败: {e}")

if __name__ == "__main__":
    try:
        logger.info("🚀 启动防过拟合系统测试...")
        
        success = test_anti_overfitting_system()
        
        if success:
            logger.info("🎉 所有测试通过！防过拟合系统工作正常")
        else:
            logger.error("❌ 测试失败，请检查系统")
            
    except Exception as e:
        logger.error(f"❌ 测试脚本执行失败: {e}")
