import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay

def plot_loss(losses):
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('training_loss.png',dpi=300)

def plot_confusion(cm):
    disp=ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig('confusion_matrix.png',dpi=300)
