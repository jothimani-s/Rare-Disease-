import shap
import torch
import matplotlib.pyplot as plt

class ExplainabilityModule:
    def __init__(self, model):
        self.model = model

    def shap_explainer(self, background, samples):
        explainer = shap.DeepExplainer(self.model, background)
        values = explainer.shap_values(samples)
        return values

    def feature_importance_plot(self, shap_values, samples):
        shap.summary_plot(shap_values, samples, show=False)
        plt.savefig('shap_summary.png', dpi=300)

    def attention_heatmap(self, attention_weights):
        plt.figure(figsize=(8,6))
        plt.imshow(attention_weights.detach().cpu(), aspect='auto')
        plt.colorbar()
        plt.title("Attention Heatmap")
        plt.savefig("attention_heatmap.png", dpi=300)
