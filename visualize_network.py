"""
YOLOv11 Network Architecture Visualization
==========================================
Creates a clean, visual representation of the neural network architecture
showing the most important layers and connections.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO


def visualize_yolo_architecture(model_path='best.pt'):
    """Create a detailed visualization of YOLOv11 architecture"""
    
    print("Loading model...")
    model = YOLO(model_path)
    pt_model = model.model
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 30)
    ax.axis('off')
    
    # Title
    fig.suptitle('YOLOv11 Nano Architecture - Pothole & Crack Detection', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    # Color scheme for different layer types
    colors = {
        'conv': '#3498db',      # Blue
        'c2f': '#e74c3c',       # Red
        'c2psa': '#9b59b6',     # Purple
        'sppf': '#f39c12',      # Orange
        'upsample': '#1abc9c',  # Turquoise
        'concat': '#95a5a6',    # Gray
        'detect': '#2ecc71'     # Green
    }
    
    # Define the network structure with key layers
    layers_info = [
        # Backbone
        {'name': 'Input', 'type': 'input', 'shape': '[3, 640, 640]', 'pos': (2, 28), 'params': 0},
        {'name': 'Conv 0', 'type': 'conv', 'shape': '[16, 320, 320]', 'pos': (2, 26), 'params': 432},
        {'name': 'Conv 1', 'type': 'conv', 'shape': '[32, 160, 160]', 'pos': (2, 24), 'params': 4608},
        {'name': 'C2f 2', 'type': 'c2f', 'shape': '[32, 160, 160]', 'pos': (2, 22), 'params': 7360},
        {'name': 'Conv 3', 'type': 'conv', 'shape': '[64, 80, 80]', 'pos': (2, 20), 'params': 36864},
        {'name': 'C2f 4', 'type': 'c2f', 'shape': '[64, 80, 80]', 'pos': (2, 18), 'params': 29440, 'save': True},
        {'name': 'Conv 5', 'type': 'conv', 'shape': '[128, 40, 40]', 'pos': (2, 16), 'params': 147456},
        {'name': 'C2PSA 6', 'type': 'c2psa', 'shape': '[128, 40, 40]', 'pos': (2, 14), 'params': 117248, 'save': True},
        {'name': 'Conv 7', 'type': 'conv', 'shape': '[256, 20, 20]', 'pos': (2, 12), 'params': 294912},
        {'name': 'C2PSA 8', 'type': 'c2psa', 'shape': '[256, 20, 20]', 'pos': (2, 10), 'params': 452864},
        {'name': 'SPPF 9', 'type': 'sppf', 'shape': '[256, 20, 20]', 'pos': (2, 8), 'params': 328448, 'save': True},
        
        # Neck - FPN
        {'name': 'C2PSA 10', 'type': 'c2psa', 'shape': '[256, 20, 20]', 'pos': (6, 8), 'params': 330752},
        {'name': 'Upsample 11', 'type': 'upsample', 'shape': '[256, 40, 40]', 'pos': (6, 10), 'params': 0},
        {'name': 'Concat 12', 'type': 'concat', 'shape': '[384, 40, 40]', 'pos': (8, 12), 'params': 0},
        {'name': 'C2f 13', 'type': 'c2f', 'shape': '[128, 40, 40]', 'pos': (10, 12), 'params': 111232, 'save': True},
        {'name': 'Upsample 14', 'type': 'upsample', 'shape': '[128, 80, 80]', 'pos': (10, 14), 'params': 0},
        {'name': 'Concat 15', 'type': 'concat', 'shape': '[192, 80, 80]', 'pos': (12, 16), 'params': 0},
        {'name': 'C2f 16', 'type': 'c2f', 'shape': '[64, 80, 80]', 'pos': (14, 16), 'params': 41600, 'save': True},
        
        # Neck - PAN
        {'name': 'Conv 17', 'type': 'conv', 'shape': '[64, 40, 40]', 'pos': (14, 14), 'params': 36864},
        {'name': 'Concat 18', 'type': 'concat', 'shape': '[192, 40, 40]', 'pos': (12, 12), 'params': 0},
        {'name': 'C2f 19', 'type': 'c2f', 'shape': '[128, 40, 40]', 'pos': (10, 10), 'params': 123008, 'save': True},
        {'name': 'Conv 20', 'type': 'conv', 'shape': '[128, 20, 20]', 'pos': (10, 8), 'params': 147456},
        {'name': 'Concat 21', 'type': 'concat', 'shape': '[384, 20, 20]', 'pos': (8, 6), 'params': 0},
        {'name': 'C2PSA 22', 'type': 'c2psa', 'shape': '[256, 20, 20]', 'pos': (6, 4), 'params': 452864, 'save': True},
        
        # Head - Detection
        {'name': 'Detect 23', 'type': 'detect', 'shape': '3 scales', 'pos': (14, 2), 'params': 586598},
        {'name': 'Output P3/8', 'type': 'detect', 'shape': '[64, 80, 80]', 'pos': (16, 4), 'params': 0},
        {'name': 'Output P4/16', 'type': 'detect', 'shape': '[128, 40, 40]', 'pos': (16, 2), 'params': 0},
        {'name': 'Output P5/32', 'type': 'detect', 'shape': '[256, 20, 20]', 'pos': (16, 0.5), 'params': 0},
    ]
    
    # Draw layers
    layer_boxes = {}
    for layer in layers_info:
        x, y = layer['pos']
        layer_type = layer['type']
        
        # Determine color and size
        if layer_type == 'input':
            color = '#34495e'
            width, height = 1.5, 0.8
        elif layer_type == 'detect':
            color = colors.get(layer_type, '#95a5a6')
            width, height = 2.0, 0.6
        elif layer_type in ['c2f', 'c2psa']:
            color = colors.get(layer_type, '#95a5a6')
            width, height = 1.8, 1.0
        elif layer_type == 'sppf':
            color = colors.get(layer_type, '#95a5a6')
            width, height = 1.8, 0.9
        elif layer_type in ['upsample', 'concat']:
            color = colors.get(layer_type, '#95a5a6')
            width, height = 1.2, 0.5
        else:
            color = colors.get(layer_type, '#95a5a6')
            width, height = 1.5, 0.7
        
        # Create box
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.05",
            edgecolor='black',
            facecolor=color,
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(box)
        layer_boxes[layer['name']] = (x, y)
        
        # Add text
        ax.text(x, y + 0.15, layer['name'], 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        ax.text(x, y - 0.05, layer['shape'], 
                ha='center', va='center', fontsize=7, color='white')
        
        # Add parameter count for important layers
        if layer['params'] > 1000:
            params_text = f"{layer['params']:,}" if layer['params'] < 1000000 else f"{layer['params']/1000:.1f}K"
            ax.text(x, y - 0.25, params_text, 
                    ha='center', va='center', fontsize=6, color='yellow', fontweight='bold')
        
        # Add marker for saved outputs
        if layer.get('save', False):
            ax.plot(x + width/2 + 0.1, y, 'r*', markersize=12)
    
    # Draw connections (main flow)
    main_flow = [
        ('Input', 'Conv 0'),
        ('Conv 0', 'Conv 1'),
        ('Conv 1', 'C2f 2'),
        ('C2f 2', 'Conv 3'),
        ('Conv 3', 'C2f 4'),
        ('C2f 4', 'Conv 5'),
        ('Conv 5', 'C2PSA 6'),
        ('C2PSA 6', 'Conv 7'),
        ('Conv 7', 'C2PSA 8'),
        ('C2PSA 8', 'SPPF 9'),
        ('SPPF 9', 'C2PSA 10'),
        ('C2PSA 10', 'Upsample 11'),
        ('Upsample 11', 'Concat 12'),
        ('Concat 12', 'C2f 13'),
        ('C2f 13', 'Upsample 14'),
        ('Upsample 14', 'Concat 15'),
        ('Concat 15', 'C2f 16'),
        ('C2f 16', 'Conv 17'),
        ('Conv 17', 'Concat 18'),
        ('Concat 18', 'C2f 19'),
        ('C2f 19', 'Conv 20'),
        ('Conv 20', 'Concat 21'),
        ('Concat 21', 'C2PSA 22'),
    ]
    
    # Draw skip connections
    skip_connections = [
        ('C2PSA 6', 'Concat 12'),
        ('C2f 4', 'Concat 15'),
        ('C2f 13', 'Concat 18'),
        ('C2PSA 10', 'Concat 21'),
    ]
    
    # Draw detection head connections
    detect_connections = [
        ('C2f 16', 'Detect 23'),
        ('C2f 19', 'Detect 23'),
        ('C2PSA 22', 'Detect 23'),
        ('Detect 23', 'Output P3/8'),
        ('Detect 23', 'Output P4/16'),
        ('Detect 23', 'Output P5/32'),
    ]
    
    # Draw main flow arrows
    for src, dst in main_flow:
        if src in layer_boxes and dst in layer_boxes:
            x1, y1 = layer_boxes[src]
            x2, y2 = layer_boxes[dst]
            arrow = FancyArrowPatch(
                (x1, y1 - 0.4), (x2, y2 + 0.4),
                arrowstyle='->,head_width=0.3,head_length=0.3',
                color='black',
                linewidth=2,
                alpha=0.7
            )
            ax.add_patch(arrow)
    
    # Draw skip connections
    for src, dst in skip_connections:
        if src in layer_boxes and dst in layer_boxes:
            x1, y1 = layer_boxes[src]
            x2, y2 = layer_boxes[dst]
            arrow = FancyArrowPatch(
                (x1 + 0.9, y1), (x2 - 0.6, y2),
                arrowstyle='->,head_width=0.3,head_length=0.3',
                color='red',
                linewidth=2,
                linestyle='--',
                alpha=0.6
            )
            ax.add_patch(arrow)
    
    # Draw detection connections
    for src, dst in detect_connections:
        if src in layer_boxes and dst in layer_boxes:
            x1, y1 = layer_boxes[src]
            x2, y2 = layer_boxes[dst]
            arrow = FancyArrowPatch(
                (x1, y1 - 0.4), (x2, y2 + 0.3),
                arrowstyle='->,head_width=0.3,head_length=0.3',
                color='green',
                linewidth=2.5,
                alpha=0.7
            )
            ax.add_patch(arrow)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['conv'], edgecolor='black', label='Conv Layer'),
        mpatches.Patch(facecolor=colors['c2f'], edgecolor='black', label='C2f Block'),
        mpatches.Patch(facecolor=colors['c2psa'], edgecolor='black', label='C2PSA (Attention)'),
        mpatches.Patch(facecolor=colors['sppf'], edgecolor='black', label='SPPF (Pooling)'),
        mpatches.Patch(facecolor=colors['upsample'], edgecolor='black', label='Upsample'),
        mpatches.Patch(facecolor=colors['concat'], edgecolor='black', label='Concatenation'),
        mpatches.Patch(facecolor=colors['detect'], edgecolor='black', label='Detection Head'),
        mpatches.Patch(facecolor='white', edgecolor='red', linestyle='--', label='Skip Connection'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Main Flow'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add annotations
    ax.text(1, 29, 'Backbone\n(Feature Extraction)', 
            fontsize=12, fontweight='bold', ha='left', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(7, 13, 'Neck - FPN\n(Top-Down)', 
            fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(11, 9, 'Neck - PAN\n(Bottom-Up)', 
            fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(17, 2.5, 'Detection\nHead', 
            fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Add model statistics
    stats_text = (
        'Model Statistics:\n'
        f'Total Parameters: 2,590,230\n'
        f'Model Size: 9.88 MB\n'
        f'Input: 640×640 RGB\n'
        f'Output: 3 scales (P3, P4, P5)\n'
        f'Classes: 2 (Pothole, Crack)\n'
        f'GFLOPs: 6.4'
    )
    ax.text(0.5, 2, stats_text, 
            fontsize=10, ha='left', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            family='monospace')
    
    # Add red star explanation
    ax.text(18, 29, '★ = Saved for skip connections', 
            fontsize=9, ha='right', color='red', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'model_analysis_output/network_architecture_visualization.png'
    os.makedirs('model_analysis_output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Network visualization saved to: {output_path}")
    
    # Also create a simplified version
    create_simplified_visualization(model, model_path)
    
    plt.show()


def create_simplified_visualization(model, model_path):
    """Create a simplified block diagram"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    fig.suptitle('YOLOv11 Simplified Architecture Overview', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Define blocks
    blocks = [
        {'name': 'Input Image\n640×640×3', 'pos': (2, 10), 'color': '#34495e', 'size': (2, 1.5)},
        {'name': 'Stem\n2 Conv layers\n↓ 4×', 'pos': (2, 8), 'color': '#3498db', 'size': (2, 1.2)},
        {'name': 'Stage 1\nC2f Block\n32 channels', 'pos': (2, 6.3), 'color': '#e74c3c', 'size': (2, 1.2)},
        {'name': 'Stage 2\nC2f Block\n64 channels\n↓ 2×', 'pos': (2, 4.6), 'color': '#e74c3c', 'size': (2, 1.2)},
        {'name': 'Stage 3\nC2PSA + Attention\n128 channels\n↓ 2×', 'pos': (2, 2.9), 'color': '#9b59b6', 'size': (2, 1.2)},
        {'name': 'Stage 4\nC2PSA + SPPF\n256 channels\n↓ 2×', 'pos': (2, 1.2), 'color': '#9b59b6', 'size': (2, 1.2)},
        
        # Neck
        {'name': 'FPN\nTop-Down\n3 levels', 'pos': (6, 3.5), 'color': '#1abc9c', 'size': (2.5, 1.5)},
        {'name': 'PAN\nBottom-Up\n3 levels', 'pos': (6, 1.5), 'color': '#16a085', 'size': (2.5, 1.5)},
        
        # Head
        {'name': 'Detection Head\nP3 (80×80)\nSmall Objects', 'pos': (10, 4.5), 'color': '#27ae60', 'size': (2.5, 1.2)},
        {'name': 'Detection Head\nP4 (40×40)\nMedium Objects', 'pos': (10, 2.5), 'color': '#27ae60', 'size': (2.5, 1.2)},
        {'name': 'Detection Head\nP5 (20×20)\nLarge Objects', 'pos': (10, 0.5), 'color': '#27ae60', 'size': (2.5, 1.2)},
        
        # Output
        {'name': 'Predictions\nClass: Pothole/Crack\nBBox + Confidence', 'pos': (14, 2.5), 'color': '#2ecc71', 'size': (2, 2.5)},
    ]
    
    # Draw blocks
    for block in blocks:
        x, y = block['pos']
        w, h = block['size']
        
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=block['color'],
            linewidth=2.5,
            alpha=0.85
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(x, y, block['name'], 
                ha='center', va='center', fontsize=9, fontweight='bold', 
                color='white', linespacing=1.5)
    
    # Draw connections
    connections = [
        ((2, 9.25), (2, 8.6)),    # Input to Stem
        ((2, 7.4), (2, 6.9)),     # Stem to Stage 1
        ((2, 5.7), (2, 5.2)),     # Stage 1 to 2
        ((2, 4.0), (2, 3.5)),     # Stage 2 to 3
        ((2, 2.3), (2, 1.8)),     # Stage 3 to 4
        ((3.0, 1.2), (4.75, 2.75)),   # Stage 4 to FPN
        ((6, 2.75), (6, 2.25)),   # FPN to PAN
        ((7.25, 4.25), (8.75, 4.5)),  # To Head P3
        ((7.25, 1.5), (8.75, 2.5)),   # To Head P4
        ((7.25, 0.75), (8.75, 0.5)),  # To Head P5
        ((11.25, 4.5), (13, 3.75)),   # Head P3 to Output
        ((11.25, 2.5), (13, 2.5)),    # Head P4 to Output
        ((11.25, 0.5), (13, 1.25)),   # Head P5 to Output
    ]
    
    for (x1, y1), (x2, y2) in connections:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->,head_width=0.4,head_length=0.4',
            color='black',
            linewidth=3,
            alpha=0.7
        )
        ax.add_patch(arrow)
    
    # Add skip connections
    skip_conns = [
        ((3, 4.6), (4.75, 4.25), 'P3'),
        ((3, 2.9), (4.75, 3.5), 'P4'),
        ((3, 1.2), (4.75, 2.75), 'P5'),
    ]
    
    for (x1, y1), (x2, y2), label in skip_conns:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->,head_width=0.3,head_length=0.3',
            color='red',
            linewidth=2.5,
            linestyle='--',
            alpha=0.6
        )
        ax.add_patch(arrow)
    
    # Add labels
    ax.text(2, 11, 'BACKBONE', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(6, 5.5, 'NECK', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(10, 5.8, 'MULTI-SCALE HEAD', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Add key features
    features_text = (
        'Key Features:\n'
        '• C2PSA: Context-Spatial Attention\n'
        '• SPPF: Spatial Pyramid Pooling\n'
        '• FPN: Feature Pyramid Network\n'
        '• PAN: Path Aggregation Network\n'
        '• Multi-scale detection (3 scales)\n'
        '• Anchor-free detection'
    )
    ax.text(0.5, 8, features_text, fontsize=9, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            family='monospace', linespacing=1.8)
    
    # Add parameters info
    params_text = (
        'Model Info:\n'
        'Parameters: 2.59M\n'
        'Size: 9.88 MB\n'
        'GFLOPs: 6.4\n'
        'Speed: Real-time\n'
        'Classes: 2\n'
        'mAP@0.5: 74.95%'
    )
    ax.text(15.5, 8, params_text, fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
            family='monospace', linespacing=1.8)
    
    plt.tight_layout()
    
    # Save
    output_path = 'model_analysis_output/network_simplified_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Simplified diagram saved to: {output_path}")


def main():
    """Main function"""
    print("="*80)
    print("YOLOv11 NETWORK ARCHITECTURE VISUALIZATION")
    print("="*80)
    
    # Check for model
    model_files = ['best.pt', 'final_model.pt', 'last.pt']
    model_path = None
    
    for mf in model_files:
        if os.path.exists(mf):
            model_path = mf
            break
    
    if not model_path:
        print("❌ No model file found!")
        return
    
    print(f"\nUsing model: {model_path}")
    print("\nGenerating network visualizations...")
    print("-" * 80)
    
    visualize_yolo_architecture(model_path)
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. network_architecture_visualization.png - Detailed layer-by-layer view")
    print("  2. network_simplified_diagram.png - High-level block diagram")
    print("\nFiles saved in: model_analysis_output/")
    print("="*80)


if __name__ == "__main__":
    main()
