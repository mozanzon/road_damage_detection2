"""
YOLOv11 Model Analysis & Visualization
======================================
This script provides comprehensive analysis of the trained YOLOv11 model including:
- Model architecture details
- Layer-by-layer parameters
- Training hyperparameters
- Model statistics and visualization
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Try to import required libraries
try:
    from ultralytics import YOLO
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "pandas", "matplotlib", "seaborn"])
    from ultralytics import YOLO
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns


class ModelAnalyzer:
    """Comprehensive model analysis and visualization"""
    
    def __init__(self, model_path='best.pt'):
        """Initialize the analyzer with model path"""
        self.model_path = model_path
        self.model = None
        self.model_info = {}
        
        print("="*80)
        print("YOLOv11 MODEL ANALYSIS & VISUALIZATION")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model Path: {model_path}")
        print("="*80)
        
    def load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(self.model_path):
                print(f"\n❌ Model file not found: {self.model_path}")
                print("\nAvailable model files:")
                for model_file in ['best.pt', 'final_model.pt', 'last.pt']:
                    if os.path.exists(model_file):
                        print(f"  ✓ {model_file}")
                        self.model_path = model_file
                        break
            
            print(f"\n📦 Loading model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def analyze_training_config(self):
        """Analyze training configuration from args.yaml"""
        print("\n" + "="*80)
        print("TRAINING CONFIGURATION")
        print("="*80)
        
        try:
            if os.path.exists('args.yaml'):
                with open('args.yaml', 'r') as f:
                    config = yaml.safe_load(f)
                
                # Organize parameters by category
                categories = {
                    'Model Configuration': ['task', 'model', 'pretrained', 'imgsz'],
                    'Training Parameters': ['epochs', 'batch', 'patience', 'workers', 'device'],
                    'Optimization': ['optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay'],
                    'Warmup Settings': ['warmup_epochs', 'warmup_momentum', 'warmup_bias_lr'],
                    'Loss Weights': ['box', 'cls', 'dfl'],
                    'Data Augmentation': ['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 
                                         'scale', 'shear', 'perspective', 'flipud', 'fliplr',
                                         'mosaic', 'mixup', 'cutmix', 'copy_paste'],
                    'Validation': ['val', 'iou', 'max_det', 'conf'],
                }
                
                for category, keys in categories.items():
                    print(f"\n📋 {category}:")
                    print("-" * 80)
                    for key in keys:
                        if key in config:
                            value = config[key]
                            print(f"  {key:25s}: {value}")
                
                self.model_info['training_config'] = config
                return config
            else:
                print("⚠️  args.yaml not found")
                return None
        except Exception as e:
            print(f"❌ Error reading training config: {str(e)}")
            return None
    
    def analyze_dataset_config(self):
        """Analyze dataset configuration"""
        print("\n" + "="*80)
        print("DATASET CONFIGURATION")
        print("="*80)
        
        try:
            data_yaml_path = 'dataset/data.yaml'
            if os.path.exists(data_yaml_path):
                with open(data_yaml_path, 'r') as f:
                    dataset_config = yaml.safe_load(f)
                
                print(f"\n📊 Dataset Information:")
                print("-" * 80)
                print(f"  Number of Classes    : {dataset_config.get('nc', 'N/A')}")
                print(f"  Class Names          : {dataset_config.get('names', 'N/A')}")
                print(f"  Train Path           : {dataset_config.get('train', 'N/A')}")
                print(f"  Validation Path      : {dataset_config.get('val', 'N/A')}")
                print(f"  Test Path            : {dataset_config.get('test', 'N/A')}")
                
                # Count images in each split
                dataset_base = Path('dataset')
                if dataset_base.exists():
                    print(f"\n📁 Dataset Statistics:")
                    print("-" * 80)
                    for split in ['train', 'val', 'test']:
                        img_path = dataset_base / 'images' / split
                        lbl_path = dataset_base / 'labels' / split
                        if img_path.exists():
                            img_count = len(list(img_path.glob('*.jpg'))) + len(list(img_path.glob('*.png')))
                            lbl_count = len(list(lbl_path.glob('*.txt'))) if lbl_path.exists() else 0
                            print(f"  {split.upper():12s}: {img_count:5d} images, {lbl_count:5d} labels")
                
                self.model_info['dataset_config'] = dataset_config
                return dataset_config
            else:
                print(f"⚠️  Dataset config not found at: {data_yaml_path}")
                return None
        except Exception as e:
            print(f"❌ Error reading dataset config: {str(e)}")
            return None
    
    def analyze_model_architecture(self):
        """Analyze model architecture in detail"""
        print("\n" + "="*80)
        print("MODEL ARCHITECTURE")
        print("="*80)
        
        try:
            # Get model info
            model_info = self.model.info(detailed=True, verbose=True)
            
            # Access the underlying PyTorch model
            pt_model = self.model.model
            
            print(f"\n🏗️  Model Structure:")
            print("-" * 80)
            
            # Count parameters by type
            total_params = 0
            trainable_params = 0
            
            for name, param in pt_model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            print(f"  Total Parameters     : {total_params:,}")
            print(f"  Trainable Parameters : {trainable_params:,}")
            print(f"  Non-trainable Params : {total_params - trainable_params:,}")
            print(f"  Model Size (MB)      : {total_params * 4 / (1024**2):.2f}")  # Assuming float32
            
            # Layer-by-layer analysis
            print(f"\n📐 Layer-by-Layer Breakdown:")
            print("-" * 80)
            print(f"{'Layer':<40} {'Type':<25} {'Parameters':>15}")
            print("-" * 80)
            
            layer_summary = []
            for i, (name, module) in enumerate(pt_model.named_modules()):
                if len(list(module.children())) == 0:  # Leaf modules only
                    params = sum(p.numel() for p in module.parameters())
                    if params > 0:
                        module_type = module.__class__.__name__
                        layer_summary.append({
                            'Layer': name if name else f'layer_{i}',
                            'Type': module_type,
                            'Parameters': params
                        })
                        print(f"{name if name else f'layer_{i}':<40} {module_type:<25} {params:>15,}")
            
            self.model_info['architecture'] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'layer_summary': layer_summary
            }
            
            return layer_summary
            
        except Exception as e:
            print(f"❌ Error analyzing architecture: {str(e)}")
            return None
    
    def analyze_model_performance(self):
        """Analyze model performance metrics"""
        print("\n" + "="*80)
        print("MODEL PERFORMANCE METRICS")
        print("="*80)
        
        try:
            # Check if results.csv exists
            if os.path.exists('results.csv'):
                df = pd.read_csv('results.csv')
                df.columns = df.columns.str.strip()  # Remove whitespace from column names
                
                print(f"\n📊 Training Results Summary:")
                print("-" * 80)
                
                # Get final epoch metrics
                if len(df) > 0:
                    final_metrics = df.iloc[-1]
                    
                    print(f"  Total Epochs Trained : {len(df)}")
                    print(f"\n  Final Epoch Metrics:")
                    for col in df.columns:
                        if col.strip():  # Skip empty columns
                            try:
                                value = final_metrics[col]
                                print(f"    {col:30s}: {value:.6f}" if isinstance(value, (int, float)) else f"    {col:30s}: {value}")
                            except:
                                pass
                    
                    # Best metrics
                    print(f"\n  Best Metrics Across Training:")
                    metrics_to_track = ['metrics/precision(B)', 'metrics/recall(B)', 
                                       'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
                    
                    for metric in metrics_to_track:
                        if metric in df.columns:
                            best_value = df[metric].max()
                            best_epoch = df[metric].idxmax() + 1
                            print(f"    {metric:30s}: {best_value:.6f} (epoch {best_epoch})")
                    
                    self.model_info['performance'] = df.to_dict('records')[-1]
                    return df
            else:
                print("⚠️  results.csv not found")
                return None
                
        except Exception as e:
            print(f"❌ Error analyzing performance: {str(e)}")
            return None
    
    def visualize_model(self):
        """Create visualizations of model characteristics"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        try:
            # Create output directory
            output_dir = Path('model_analysis_output')
            output_dir.mkdir(exist_ok=True)
            
            # 1. Parameter Distribution
            if 'architecture' in self.model_info and self.model_info['architecture']:
                layer_summary = self.model_info['architecture']['layer_summary']
                if layer_summary and len(layer_summary) > 0:
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    fig.suptitle('YOLOv11 Model Analysis', fontsize=16, fontweight='bold')
                    
                    # Plot 1: Top 20 layers by parameters
                    df_layers = pd.DataFrame(layer_summary)
                    df_layers_sorted = df_layers.nlargest(20, 'Parameters')
                    
                    ax = axes[0, 0]
                    bars = ax.barh(range(len(df_layers_sorted)), df_layers_sorted['Parameters'])
                    ax.set_yticks(range(len(df_layers_sorted)))
                    ax.set_yticklabels([name[:30] for name in df_layers_sorted['Layer']], fontsize=8)
                    ax.set_xlabel('Number of Parameters', fontsize=10)
                    ax.set_title('Top 20 Layers by Parameter Count', fontsize=12, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    
                    # Color bars by magnitude
                    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                    for bar, color in zip(bars, colors):
                        bar.set_color(color)
                    
                    # Plot 2: Layer type distribution
                    ax = axes[0, 1]
                    type_counts = df_layers['Type'].value_counts()
                    colors = plt.cm.Set3(range(len(type_counts)))
                    wedges, texts, autotexts = ax.pie(type_counts.values, labels=type_counts.index, 
                                                        autopct='%1.1f%%', colors=colors, startangle=90)
                    for text in texts:
                        text.set_fontsize(9)
                    for autotext in autotexts:
                        autotext.set_fontsize(8)
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    ax.set_title('Layer Type Distribution', fontsize=12, fontweight='bold')
                    
                    # Plot 3: Parameter statistics
                    ax = axes[1, 0]
                    total_params = self.model_info['architecture']['total_params']
                    trainable_params = self.model_info['architecture']['trainable_params']
                    non_trainable = total_params - trainable_params
                    
                    categories = ['Total\nParameters', 'Trainable\nParameters', 'Non-trainable\nParameters']
                    values = [total_params, trainable_params, non_trainable]
                    colors = ['#3498db', '#2ecc71', '#e74c3c']
                    
                    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                    ax.set_ylabel('Number of Parameters', fontsize=10)
                    ax.set_title('Parameter Statistics', fontsize=12, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:,}',
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
                    # Plot 4: Model summary text
                    ax = axes[1, 1]
                    ax.axis('off')
                    
                    summary_text = "MODEL SUMMARY\n" + "="*40 + "\n\n"
                    summary_text += f"Model: YOLOv11 Nano\n"
                    summary_text += f"Task: Object Detection\n\n"
                    summary_text += f"Total Parameters: {total_params:,}\n"
                    summary_text += f"Trainable: {trainable_params:,}\n"
                    summary_text += f"Model Size: {total_params * 4 / (1024**2):.2f} MB\n\n"
                    
                    if 'training_config' in self.model_info:
                        config = self.model_info['training_config']
                        summary_text += f"Training Configuration:\n"
                        summary_text += f"  Epochs: {config.get('epochs', 'N/A')}\n"
                        summary_text += f"  Batch Size: {config.get('batch', 'N/A')}\n"
                        summary_text += f"  Image Size: {config.get('imgsz', 'N/A')}\n"
                        summary_text += f"  Learning Rate: {config.get('lr0', 'N/A')}\n"
                        summary_text += f"  Optimizer: {config.get('optimizer', 'N/A')}\n\n"
                    
                    if 'dataset_config' in self.model_info:
                        dataset = self.model_info['dataset_config']
                        summary_text += f"Dataset:\n"
                        summary_text += f"  Classes: {dataset.get('nc', 'N/A')}\n"
                        summary_text += f"  Names: {', '.join(dataset.get('names', []))}\n"
                    
                    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                           fontsize=10, verticalalignment='top',
                           fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
                    
                    plt.tight_layout()
                    output_path = output_dir / 'model_architecture_analysis.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    print(f"✅ Saved architecture visualization: {output_path}")
                    plt.close()
            
            # 2. Training metrics visualization
            if os.path.exists('results.csv'):
                df = pd.read_csv('results.csv')
                df.columns = df.columns.str.strip()
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle('Training Performance Metrics', fontsize=16, fontweight='bold')
                
                epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
                
                # Loss plots
                loss_cols = [col for col in df.columns if 'loss' in col.lower()]
                if loss_cols:
                    ax = axes[0, 0]
                    for col in loss_cols:
                        ax.plot(epochs, df[col], marker='o', markersize=3, label=col, linewidth=2)
                    ax.set_xlabel('Epoch', fontsize=10)
                    ax.set_ylabel('Loss', fontsize=10)
                    ax.set_title('Training & Validation Losses', fontsize=12, fontweight='bold')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                
                # Precision & Recall
                ax = axes[0, 1]
                if 'metrics/precision(B)' in df.columns:
                    ax.plot(epochs, df['metrics/precision(B)'], marker='o', markersize=3, 
                           label='Precision', linewidth=2, color='blue')
                if 'metrics/recall(B)' in df.columns:
                    ax.plot(epochs, df['metrics/recall(B)'], marker='s', markersize=3, 
                           label='Recall', linewidth=2, color='green')
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel('Score', fontsize=10)
                ax.set_title('Precision & Recall', fontsize=12, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1])
                
                # mAP metrics
                ax = axes[1, 0]
                if 'metrics/mAP50(B)' in df.columns:
                    ax.plot(epochs, df['metrics/mAP50(B)'], marker='o', markersize=3, 
                           label='mAP@0.5', linewidth=2, color='red')
                if 'metrics/mAP50-95(B)' in df.columns:
                    ax.plot(epochs, df['metrics/mAP50-95(B)'], marker='s', markersize=3, 
                           label='mAP@0.5:0.95', linewidth=2, color='orange')
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel('mAP', fontsize=10)
                ax.set_title('Mean Average Precision', fontsize=12, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1])
                
                # Final metrics summary
                ax = axes[1, 1]
                ax.axis('off')
                
                final_metrics = df.iloc[-1]
                metrics_text = "FINAL METRICS\n" + "="*40 + "\n\n"
                
                key_metrics = [
                    ('metrics/precision(B)', 'Precision'),
                    ('metrics/recall(B)', 'Recall'),
                    ('metrics/mAP50(B)', 'mAP@0.5'),
                    ('metrics/mAP50-95(B)', 'mAP@0.5:0.95'),
                ]
                
                for col, label in key_metrics:
                    if col in df.columns:
                        value = final_metrics[col]
                        best_value = df[col].max()
                        best_epoch = df[col].idxmax() + 1
                        metrics_text += f"{label}:\n"
                        metrics_text += f"  Final: {value:.4f}\n"
                        metrics_text += f"  Best: {best_value:.4f} (epoch {best_epoch})\n\n"
                
                ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
                       fontsize=11, verticalalignment='top',
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
                
                plt.tight_layout()
                output_path = output_dir / 'training_metrics.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✅ Saved training metrics visualization: {output_path}")
                plt.close()
            
            print(f"\n✅ All visualizations saved to: {output_dir}")
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def export_summary_report(self):
        """Export a comprehensive summary report"""
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)
        
        try:
            output_dir = Path('model_analysis_output')
            output_dir.mkdir(exist_ok=True)
            
            report_path = output_dir / 'model_summary_report.txt'
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("YOLOV11 MODEL ANALYSIS REPORT\n")
                f.write("="*80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model File: {self.model_path}\n")
                f.write("="*80 + "\n\n")
                
                # Model Architecture
                f.write("1. MODEL ARCHITECTURE\n")
                f.write("-"*80 + "\n")
                if 'architecture' in self.model_info:
                    arch = self.model_info['architecture']
                    f.write(f"Total Parameters: {arch['total_params']:,}\n")
                    f.write(f"Trainable Parameters: {arch['trainable_params']:,}\n")
                    f.write(f"Model Size: {arch['total_params'] * 4 / (1024**2):.2f} MB\n\n")
                    
                    f.write("Layer Summary (Top 20):\n")
                    layer_summary = sorted(arch['layer_summary'], key=lambda x: x['Parameters'], reverse=True)[:20]
                    for i, layer in enumerate(layer_summary, 1):
                        f.write(f"  {i:2d}. {layer['Layer'][:40]:40s} {layer['Type']:25s} {layer['Parameters']:>15,}\n")
                f.write("\n")
                
                # Training Configuration
                f.write("2. TRAINING CONFIGURATION\n")
                f.write("-"*80 + "\n")
                if 'training_config' in self.model_info:
                    config = self.model_info['training_config']
                    for key, value in sorted(config.items()):
                        f.write(f"  {key:30s}: {value}\n")
                f.write("\n")
                
                # Dataset Configuration
                f.write("3. DATASET CONFIGURATION\n")
                f.write("-"*80 + "\n")
                if 'dataset_config' in self.model_info:
                    dataset = self.model_info['dataset_config']
                    for key, value in dataset.items():
                        f.write(f"  {key:30s}: {value}\n")
                f.write("\n")
                
                # Performance Metrics
                f.write("4. PERFORMANCE METRICS\n")
                f.write("-"*80 + "\n")
                if 'performance' in self.model_info:
                    perf = self.model_info['performance']
                    for key, value in sorted(perf.items()):
                        if isinstance(value, (int, float)):
                            f.write(f"  {key:30s}: {value:.6f}\n")
                        else:
                            f.write(f"  {key:30s}: {value}\n")
                f.write("\n")
                
                f.write("="*80 + "\n")
                f.write("END OF REPORT\n")
                f.write("="*80 + "\n")
            
            print(f"✅ Summary report saved: {report_path}")
            
        except Exception as e:
            print(f"❌ Error generating report: {str(e)}")
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("\n🚀 Starting complete model analysis...\n")
        
        # Load model
        if not self.load_model():
            return
        
        # Run all analyses
        self.analyze_training_config()
        self.analyze_dataset_config()
        self.analyze_model_architecture()
        self.analyze_model_performance()
        
        # Generate visualizations
        self.visualize_model()
        
        # Export report
        self.export_summary_report()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nAll results saved to: model_analysis_output/")
        print("\nGenerated files:")
        print("  - model_architecture_analysis.png")
        print("  - training_metrics.png")
        print("  - model_summary_report.txt")
        print("\n" + "="*80)


def main():
    """Main execution function"""
    # Check for model files
    model_files = ['best.pt', 'final_model.pt', 'last.pt']
    available_models = [f for f in model_files if os.path.exists(f)]
    
    if not available_models:
        print("❌ No model files found!")
        print("Please ensure one of these files exists: best.pt, final_model.pt, last.pt")
        return
    
    # Use the first available model
    model_path = available_models[0]
    
    # Create analyzer and run analysis
    analyzer = ModelAnalyzer(model_path=model_path)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
