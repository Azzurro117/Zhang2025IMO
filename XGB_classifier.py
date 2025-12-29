#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Azzurro
# Description: XGBoost Classifier for serotype prediction

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def run_simple_xgb_classifier(input_file="sample.csv", output_file="result.csv", test_size=0.2, n_estimators=1000, random_state=42, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8):
    """
    Simplified version of XGBoost classifier
    
    Args:
        input_file: Path to input CSV file (format: sample_name, serotype, feature1, feature2, ...)
        output_file: Path to save prediction probabilities
    """
    print("="*50)
    print("XGBoost Classifier - Simplified Version")
    print("="*50)
    
    # 1. Load data
    print(f"\n[Step 1] Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"  - Successfully loaded {len(df)} samples")
        print(f"  - Columns: {df.columns.tolist()}")
        print(f"  - Data shape: {df.shape}")
    except Exception as e:
        print(f"  - Error loading file: {e}")
        return
    
    # 2. Check data structure
    if df.shape[1] < 3:
        print(f"  - Error: Expected at least 3 columns, got {df.shape[1]}")
        print(f"  - Data format should be: sample_name, serotype, feature1, feature2, ...")
        return
    
    # 3. Separate training and validation data
    print(f"\n[Step 2] Separating training and validation data...")
    
    # 训练集: 分类为0-4的样本
    train_mask = df.iloc[:, 1].astype(str).isin(['0', '1', '2', '3', '4'])
    train_df = df[train_mask].copy()
    
    # 验证集: 分类为999的样本
    val_mask = df.iloc[:, 1].astype(str) == '999'
    val_df = df[val_mask].copy()
    
    print(f"  - Training samples (0-4): {len(train_df)}")
    print(f"  - Validation samples (999): {len(val_df)}")
    
    if len(train_df) == 0:
        print("  - Error: No training samples found (classes 0-4)")
        return
    
    # 4. Prepare features and labels
    print(f"\n[Step 3] Preparing features and labels...")
    
    # 训练数据
    X_train = train_df.iloc[:, 2:].values.astype(float)  # 特征: 从第三列开始
    y_train = train_df.iloc[:, 1].values.astype(int)  # 标签: 第二列
    train_sample_names = train_df.iloc[:, 0].values  # 样本名称: 第一列
    
    # 验证数据
    X_val = val_df.iloc[:, 2:].values.astype(float)  # 特征: 从第三列开始
    val_sample_names = val_df.iloc[:, 0].values  # 样本名称: 第一列
    
    print(f"  - Training features shape: {X_train.shape}")
    print(f"  - Training labels shape: {y_train.shape}")
    if len(val_df) > 0:
        print(f"  - Validation features shape: {X_val.shape}")
    else:
        print(f"  - No validation samples")
    
    # 5. Split training data into train and test sets
    print(f"\n[Step 4] Splitting training data into train/test sets...")
    
    # 检查是否有足够的样本进行分层抽样
    if len(np.unique(y_train)) > 1 and len(y_train) >= 5:
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train, y_train, 
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )
        print(f"  - Training split: {len(X_train_split)} samples")
        print(f"  - Testing split: {len(X_test_split)} samples")
    else:
        print(f"  - Warning: Not enough samples or classes for stratified split")
        print(f"  - Using simple train/test split")
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train, y_train, 
            test_size=0.2,
            random_state=42
        )
        print(f"  - Training split: {len(X_train_split)} samples")
        print(f"  - Testing split: {len(X_test_split)} samples")
    
    # 6. Scale features
    print(f"\n[Step 5] Scaling features...")
    
    # 只使用训练数据拟合scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)  # 仅使用训练集拟合
    
    # 使用相同的scaler转换测试集和验证集
    X_test_scaled = scaler.transform(X_test_split)
    
    if len(val_df) > 0:
        X_val_scaled = scaler.transform(X_val)
        print(f"  - Training features scaled: {X_train_scaled.shape}")
        print(f"  - Test features scaled: {X_test_scaled.shape}")
        print(f"  - Validation features scaled: {X_val_scaled.shape}")
    else:
        X_val_scaled = None
        print(f"  - Training features scaled: {X_train_scaled.shape}")
        print(f"  - Test features scaled: {X_test_scaled.shape}")
    
    # 7. Train model
    print(f"\n[Step 6] Training XGBoost model...")
    
    model = xgb.XGBClassifier(
        test_size=0.2,
        n_estimators=1000,
        random_state=42,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train_scaled, y_train_split)
    print(f"  - Model trained successfully")
    print(f"  - Number of classes learned: {len(model.classes_)}")
    
    # 8. Evaluate on training and test sets
    print(f"\n[Step 7] Evaluating model...")
    
    # 训练集评估
    train_pred = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train_split, train_pred)
    print(f"  - Training Accuracy: {train_accuracy:.4f}")
    
    # 测试集评估
    test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test_split, test_pred)
    print(f"  - Test Accuracy: {test_accuracy:.4f}")
    
    print(f"\n  - Classification Report (Test Set):")
    print(classification_report(y_test_split, test_pred, zero_division=0))
    
    # 9. Predict on all data (training + validation)
    print(f"\n[Step 8] Making predictions on all samples...")
    
    # 将训练集和验证集合并
    if len(val_df) > 0:
        # 首先需要将原始训练集也进行缩放
        X_all_train_scaled = scaler.transform(X_train)  # 缩放所有训练数据
        X_all = np.vstack([X_all_train_scaled, X_val_scaled])
        all_sample_names = np.concatenate([train_sample_names, val_sample_names])
        print(f"  - Total samples to predict: {len(X_all)} (training: {len(X_train)}, validation: {len(X_val)})")
    else:
        X_all = scaler.transform(X_train)  # 缩放所有训练数据
        all_sample_names = train_sample_names
        print(f"  - Total samples to predict: {len(X_all)}")
    
    # 预测概率
    all_proba = model.predict_proba(X_all)
    print(f"  - Probability matrix shape: {all_proba.shape}")
    print(f"  - Number of probability columns: {all_proba.shape[1]} (one per class)")
    
    # 10. Save results
    print(f"\n[Step 9] Saving results to {output_file}...")
    
    # 创建包含样本名称和预测概率的结果DataFrame
    # 第一列是样本名称，后面是每个类别的概率
    result_data = []
    for i, sample_name in enumerate(all_sample_names):
        row = [sample_name] + all_proba[i].tolist()
        result_data.append(row)
    
    # 创建列名
    columns = ['Sample']
    for class_idx in model.classes_:
        columns.append(f'Prob_Class_{class_idx}')
    
    result_df = pd.DataFrame(result_data, columns=columns)
    
    # 同时保存预测的类别（最大概率的类别）
    predicted_classes = model.predict(X_all)
    result_df['Predicted_Class'] = predicted_classes
    
    # 对于验证集样本，添加标记
    result_df['Is_Validation'] = False
    if len(val_df) > 0:
        val_indices = range(len(train_sample_names), len(all_sample_names))
        result_df.loc[val_indices, 'Is_Validation'] = True
    
    # 保存到CSV
    result_df.to_csv(output_file, index=False)
    
    print(f"  - Results saved successfully")
    print(f"  - Total rows saved: {len(result_df)}")
    
    # 11. 显示一些预测结果示例
    print(f"\n[Step 10] Prediction examples:")
    print("-" * 60)
    print(f"{'Sample':<10} {'Predicted':<10} {'Probabilities'}")
    print("-" * 60)
    
    # 显示前5个训练样本和后5个验证样本的预测结果
    display_count = min(5, len(result_df))
    for i in range(display_count):
        sample_name = result_df.iloc[i]['Sample']
        pred_class = result_df.iloc[i]['Predicted_Class']
        probs = result_df.iloc[i][[f'Prob_Class_{c}' for c in model.classes_]].values
        prob_str = ' '.join([f'{p:.3f}' for p in probs])
        is_val = "(val)" if result_df.iloc[i]['Is_Validation'] else ""
        print(f"{sample_name:<10} {pred_class:<10} {prob_str} {is_val}")
    
    if len(val_df) > 0 and len(result_df) > 5:
        print("...")
        for i in range(len(result_df)-min(5, len(val_df)), len(result_df)):
            sample_name = result_df.iloc[i]['Sample']
            pred_class = result_df.iloc[i]['Predicted_Class']
            probs = result_df.iloc[i][[f'Prob_Class_{c}' for c in model.classes_]].values
            prob_str = ' '.join([f'{p:.3f}' for p in probs])
            is_val = "(val)" if result_df.iloc[i]['Is_Validation'] else ""
            print(f"{sample_name:<10} {pred_class:<10} {prob_str} {is_val}")
    
    print("\n" + "="*50)
    print("Processing completed successfully!")
    print("="*50)

if __name__ == "__main__":
    # 运行分类器
    run_simple_xgb_classifier(
        input_file="sample.csv",
        output_file="result.csv",
        test_size=0.2,
        n_estimators=1000,
        random_state=42,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )
