import matplotlib.pyplot as plt
import re
import numpy as np

def parse_and_plot(file_path, output_filename="model-performance-with-avg.png"):
    """
    Parses the log file, extracts metrics including class-specific accuracies,
    calculates the average class accuracy, and plots them.
    """
    
    # 1. READ AND PARSE THE FILE
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Lists to store data
    val_steps = []
    
    val_metrics = {
        'overall_acc': [],
        'macro_f1': [],
        'class_acc': {i: [] for i in range(7)}, # 7 classes from 0 to 6
        'avg_class_acc': [] # NEW: To store (sum(class_acc) / 7) per step
    }
    
    test_metrics = {
        'overall_acc': None,
        'macro_f1': None,
        'class_acc': {},
        'avg_class_acc': None # NEW
    }

    # Split into blocks
    blocks = content.split('params:')
    
    step_counter = 0
    
    for block in blocks:
        if not block.strip(): continue
        
        # --- VALIDATION PART ---
        if 'val:' in block:
            step_counter += 1
            val_steps.append(step_counter)
            
            # Overall Accuracy
            acc_match = re.search(r'Accuracy_0:\s+([0-9\.]+)', block)
            if acc_match: val_metrics['overall_acc'].append(float(acc_match.group(1)))
            
            # Macro F1
            f1_match = re.search(r'Macro_F1:\s+([0-9\.]+)', block)
            if f1_match: val_metrics['macro_f1'].append(float(f1_match.group(1)))
            
            # Class-specific Accuracy & Average Calculation
            class_matches = re.findall(r'Class\s+(\d+):\s+([0-9\.]+)', block)
            current_step_class_accs = []
            
            for cls_id, score in class_matches:
                cls_id = int(cls_id)
                score = float(score)
                if cls_id < 7:
                    val_metrics['class_acc'][cls_id].append(score)
                    current_step_class_accs.append(score)
            
            # Calculate Average Class Accuracy for this step (Sum / 7)
            if len(current_step_class_accs) > 0:
                avg_acc = sum(current_step_class_accs) / 7.0
                val_metrics['avg_class_acc'].append(avg_acc)
            else:
                val_metrics['avg_class_acc'].append(0.0)

        # --- TEST PART ---
        if 'test:' in block:
            test_block = block.split('test:')[1]
            
            t_acc = re.search(r'Accuracy_0:\s+([0-9\.]+)', test_block)
            if t_acc: test_metrics['overall_acc'] = float(t_acc.group(1))
            
            t_f1 = re.search(r'Macro_F1:\s+([0-9\.]+)', test_block)
            if t_f1: test_metrics['macro_f1'] = float(t_f1.group(1))
            
            t_cls = re.findall(r'Class\s+(\d+):\s+([0-9\.]+)', test_block)
            test_class_vals = []
            for cls_id, score in t_cls:
                sid = int(cls_id)
                sval = float(score)
                test_metrics['class_acc'][sid] = sval
                if sid < 7: test_class_vals.append(sval)
            
            # Calculate Test Average Class Accuracy
            if len(test_class_vals) > 0:
                test_metrics['avg_class_acc'] = sum(test_class_vals) / 7.0

    # 2. PLOTTING
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
    
    # X axis
    x_axis = val_steps
    x_test = val_steps[-1] + 1 if val_steps else 1

    ax1.plot(x_axis, val_metrics['overall_acc'], marker='o', label='Acc', color='blue', linewidth=2)

    ax1.plot(x_axis, val_metrics['macro_f1'], marker='s', label='Macro F1', color='orange', linestyle='--', linewidth=2)

    ax1.plot(x_axis, val_metrics['avg_class_acc'], marker='^', label='Avg Class Acc', color='green', linewidth=2, linestyle='-.')

    if test_metrics['overall_acc']:
        ax1.plot(x_test, test_metrics['overall_acc'], marker='*', markersize=15, color='blue')
        ax1.plot([x_axis[-1], x_test], [val_metrics['overall_acc'][-1], test_metrics['overall_acc']], color='blue', alpha=0.3, linestyle=':')
        
    if test_metrics['macro_f1']:
        ax1.plot(x_test, test_metrics['macro_f1'], marker='*', markersize=15, color='orange')
        ax1.plot([x_axis[-1], x_test], [val_metrics['macro_f1'][-1], test_metrics['macro_f1']], color='orange', alpha=0.3, linestyle=':')
        
    if test_metrics['avg_class_acc']:
        ax1.plot(x_test, test_metrics['avg_class_acc'], marker='*', markersize=15, color='green')
        ax1.plot([x_axis[-1], x_test], [val_metrics['avg_class_acc'][-1], test_metrics['avg_class_acc']], color='green', alpha=0.3, linestyle=':')

    ax1.set_title("General Performance Metrics", fontsize=14)
    ax1.set_xlabel("Validation Step", fontsize=12)
    ax1.set_ylabel("Score (0-1)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=11)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for cls_id in range(7):
        if not val_metrics['class_acc'][cls_id]: continue
        
        y_vals = val_metrics['class_acc'][cls_id]
        color = colors[cls_id]
        label = f'Class {cls_id}'

        ax2.plot(x_axis, y_vals, marker='.', label=label, color=color, linewidth=1.5)

        if cls_id in test_metrics['class_acc']:
            test_val = test_metrics['class_acc'][cls_id]
            ax2.plot(x_test, test_val, marker='*', markersize=12, color=color)
            ax2.plot([x_axis[-1], x_test], [y_vals[-1], test_val], color=color, linestyle=':', alpha=0.5)

    ax2.set_title("Class-wise Accuracy Breakdown", fontsize=14)
    ax2.set_xlabel("Validation Step", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Classes", fontsize=10)
    
    plt.xticks(list(x_axis) + [x_test], list(x_axis) + ['TEST'])
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()
