from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix

def evaluate(y_true,y_pred,y_prob=None):
    results={
        'accuracy':accuracy_score(y_true,y_pred),
        'precision':precision_score(y_true,y_pred,average='macro'),
        'recall':recall_score(y_true,y_pred,average='macro'),
        'f1':f1_score(y_true,y_pred,average='macro'),
        'confusion_matrix':confusion_matrix(y_true,y_pred)
    }
    if y_prob is not None:
        try:
            results['roc_auc']=roc_auc_score(y_true,y_prob,multi_class='ovr')
        except Exception:
            pass
    return results
