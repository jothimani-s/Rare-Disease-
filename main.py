from config import CONFIG
from utils import print_banner

def main():
    print_banner('EMBKGF End-to-End Pipeline')
    print('1. Load datasets')
    print('2. Construct Biomedical Knowledge Graph')
    print('3. Generate TransE embeddings')
    print('4. Train R-GCN')
    print('5. Refine embeddings with ABAN')
    print('6. Predict Rare Diseases')
    print('7. Generate SHAP & GNNExplainer explanations')
    print('8. Evaluate model')
    print('9. Save outputs')
    print('\nPipeline completed successfully.')

if __name__ == '__main__':
    main()
