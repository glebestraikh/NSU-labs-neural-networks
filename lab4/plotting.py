import os
import numpy as np
import matplotlib.pyplot as plt

from metrics import per_class_metrics


def plot_loss_per_epoch(train_losses, test_losses, save_path, title=''):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, 'o-', label='Train loss')
    plt.plot(epochs, test_losses, 's-', label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title or 'Loss per epoch')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_accuracy_per_epoch(train_accs, test_accs, save_path, title=''):
    epochs = np.arange(1, len(train_accs) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, 'o-', label='Train accuracy')
    plt.plot(epochs, test_accs, 's-', label='Test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.02)
    plt.title(title or 'Accuracy per epoch')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

def plot_per_class_metrics(precision, recall, f1, save_path, title=''):
    n = len(precision)
    x = np.arange(n)
    width = 0.27
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1')
    plt.xticks(x)
    plt.ylim(0, 1.05)
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title(title or 'Per-class metrics')
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_sample_predictions(x_samples, y_true, y_pred, save_path,
                            n_show=12, title=''):
    n_show = min(n_show, len(x_samples))
    cols = 6
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows + 0.5))
    axes = np.atleast_2d(axes)
    for k in range(rows * cols):
        ax = axes.flat[k]
        if k < n_show:
            img = x_samples[k]
            if img.ndim == 3:
                img = img[0]
            ax.imshow(img, cmap='gray')
            ok = int(y_pred[k]) == int(y_true[k])
            color = 'green' if ok else 'red'
            ax.set_title(f'P:{int(y_pred[k])} T:{int(y_true[k])}',
                         color=color, fontsize=10)
        ax.axis('off')
    plt.suptitle(title or 'Sample test predictions')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def save_all_plots(history, y_true, y_pred, x_samples, plots_dir,
                   n_classes=10, title_prefix=''):
    os.makedirs(plots_dir, exist_ok=True)
    plot_loss_per_epoch(history['train_losses'], history['test_losses'],
                        os.path.join(plots_dir, 'loss_per_epoch.png'),
                        f'{title_prefix}Loss per epoch')
    plot_accuracy_per_epoch(history['train_accs'], history['test_accs'],
                            os.path.join(plots_dir, 'accuracy_per_epoch.png'),
                            f'{title_prefix}Accuracy per epoch')
    precision, recall, f1 = per_class_metrics(y_true, y_pred, n_classes)
    plot_per_class_metrics(precision, recall, f1,
                           os.path.join(plots_dir, 'per_class_metrics.png'),
                           f'{title_prefix}Per-class metrics')
    plot_sample_predictions(x_samples, y_true, y_pred,
                            os.path.join(plots_dir, 'sample_predictions.png'),
                            title=f'{title_prefix}Sample predictions')
    return precision, recall, f1