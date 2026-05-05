"""Analyze and visualize training results from all models"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def load_histories(results_dir='results'):
    """Load training history from all models"""
    results_dir = Path(results_dir)
    models = ['alexnet', 'vggnet', 'googlenet', 'resnet']
    histories = {}
    
    print("Loading training histories...")
    for model in models:
        history_file = results_dir / f'{model}_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                histories[model] = json.load(f)
            print(f"  ✓ {model.upper()} 히스토리 로드 완료")
        else:
            print(f"  ✗ {model.upper()} 히스토리 파일 없음")
    
    print(f"\n총 {len(histories)}개 모델 히스토리 로드됨\n")
    return histories


def plot_accuracy_curves(histories):
    """Plot training and validation accuracy curves for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('포켓몬 분류 모델 정확도 곡선', fontsize=16, fontweight='bold')
    
    models = list(histories.keys())
    colors = {'train': '#2E86AB', 'val': '#A23B72'}
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
        history = histories[model]
        epochs = range(1, len(history['train_accs']) + 1)
        
        ax = axes[idx]
        ax.plot(epochs, history['train_accs'], marker='o', label='학습 정확도', 
                color=colors['train'], linewidth=2, markersize=4)
        ax.plot(epochs, history['val_accs'], marker='s', label='검증 정확도', 
                color=colors['val'], linewidth=2, markersize=4)
        
        ax.set_xlabel('에포크')
        ax.set_ylabel('정확도')
        ax.set_title(f'{model.upper()} - 정확도', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('results/accuracy_curves.png', dpi=300, bbox_inches='tight')
    print("✓ 정확도 곡선 저장: results/accuracy_curves.png")


def plot_loss_curves(histories):
    """Plot training and validation loss curves for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('포켓몬 분류 모델 손실값 곡선', fontsize=16, fontweight='bold')
    
    models = list(histories.keys())
    colors = {'train': '#F18F01', 'val': '#C73E1D'}
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
        history = histories[model]
        epochs = range(1, len(history['train_losses']) + 1)
        
        ax = axes[idx]
        ax.plot(epochs, history['train_losses'], marker='o', label='학습 손실값', 
                color=colors['train'], linewidth=2, markersize=4)
        ax.plot(epochs, history['val_losses'], marker='s', label='검증 손실값', 
                color=colors['val'], linewidth=2, markersize=4)
        
        ax.set_xlabel('에포크')
        ax.set_ylabel('손실값 (Cross Entropy Loss)')
        ax.set_title(f'{model.upper()} - 손실값', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/loss_curves.png', dpi=300, bbox_inches='tight')
    print("✓ 손실값 곡선 저장: results/loss_curves.png")


def compare_models(histories):
    """Create comparison plots and summary table"""
    # 최종 성능 지표 추출
    performance_data = []
    
    for model in histories:
        history = histories[model]
        performance_data.append({
            'Model': model.upper(),
            'Train Accuracy': history['train_accs'][-1],
            'Val Accuracy': history['val_accs'][-1],
            'Train Loss': history['train_losses'][-1],
            'Val Loss': history['val_losses'][-1]
        })
    
    df_performance = pd.DataFrame(performance_data)
    
    # 출력
    print("\n" + "="*80)
    print("최종 모델 성능 비교")
    print("="*80)
    print(df_performance.to_string(index=False))
    print()
    
    # 최고 성능 모델
    best_model = df_performance.loc[df_performance['Val Accuracy'].idxmax()]
    print(f"🏆 최고 성능 모델: {best_model['Model']}")
    print(f"   검증 정확도: {best_model['Val Accuracy']:.4f}")
    print(f"   검증 손실값: {best_model['Val Loss']:.4f}")
    print("="*80 + "\n")
    
    # 비교 그래프
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(df_performance))
    width = 0.35
    
    # 정확도 비교
    ax1.bar(x - width/2, df_performance['Train Accuracy'], width, label='학습 정확도', color='#2E86AB')
    ax1.bar(x + width/2, df_performance['Val Accuracy'], width, label='검증 정확도', color='#A23B72')
    
    ax1.set_ylabel('정확도')
    ax1.set_title('모델별 최종 정확도 비교', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_performance['Model'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])
    
    # 손실값 비교
    ax2.bar(x - width/2, df_performance['Train Loss'], width, label='학습 손실값', color='#F18F01')
    ax2.bar(x + width/2, df_performance['Val Loss'], width, label='검증 손실값', color='#C73E1D')
    
    ax2.set_ylabel('손실값')
    ax2.set_title('모델별 최종 손실값 비교', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_performance['Model'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 모델 비교 그래프 저장: results/comparison.png")
    
    return df_performance


def plot_combined_accuracy(histories):
    """Plot all models' accuracy on single graph"""
    plt.figure(figsize=(12, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for idx, model in enumerate(histories):
        history = histories[model]
        epochs = range(1, len(history['val_accs']) + 1)
        plt.plot(epochs, history['val_accs'], marker='o', label=model.upper(), 
                color=colors[idx], linewidth=2, markersize=5)
    
    plt.xlabel('에포크', fontsize=12)
    plt.ylabel('검증 정확도', fontsize=12)
    plt.title('모든 모델 검증 정확도 비교', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig('results/combined_accuracy.png', dpi=300, bbox_inches='tight')
    print("✓ 통합 정확도 곡선 저장: results/combined_accuracy.png")


def main():
    """Main analysis function"""
    print("\n" + "="*80)
    print("포켓몬 분류 모델 결과 분석")
    print("="*80 + "\n")
    
    # 데이터 로드
    histories = load_histories()
    
    if not histories:
        print("❌ 학습 결과가 없습니다. 먼저 train.py를 실행해주세요.")
        return
    
    # 그래프 생성
    print("그래프 생성 중...")
    plot_accuracy_curves(histories)
    plot_loss_curves(histories)
    plot_combined_accuracy(histories)
    
    # 모델 비교
    print()
    compare_models(histories)
    
    print("✓ 모든 분석 완료!")
    print("  생성된 그래프들은 results/ 폴더에 저장되었습니다.\n")


if __name__ == '__main__':
    main()
