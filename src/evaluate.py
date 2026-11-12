import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,ConfusionMatrixDisplay
from .config import FIGURES, LABEL_NAMES

def evaluate(model_name,y_true,y_pred,show=True,save_fig=True):
    metrics={
        "model": model_name,
        "accuracy": accuracy_score(y_true,y_pred), #out of all tweets, how many did it get right
        "precision":precision_score(y_true,y_pred,pos_label=1,zero_division=0), #when the model says “ironic”, how often is it right
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0), #out of all the truly ironic tweets, how many did it find
        "f1": f1_score(y_true,y_pred,pos_label=1, zero_division=0),#one number that balances precision and recall, so you don’t have to look at two numbers
        "macro_f1": f1_score(y_true,y_pred,average="macro"),#it computes the F1 for each class, irony and non-irony, then averages them equally
    }

    if show:
        print(f"---{model_name}---")
        #summary table of the scores for each class
        print(classification_report(y_true,y_pred,target_names=LABEL_NAMES,digits=3))
    fig, ax = plt.subplots(figsize=(4.5, 4))

    #the 2×2 table that shows which mistakes the model makes
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=LABEL_NAMES, cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix: {model_name}")
    plt.tight_layout()

    #saves the confusion matrix as a picture file
    if save_fig:
        FIGURES.mkdir(parents=True, exist_ok=True)
        safe = model_name.lower().replace(" ", "_").replace("/", "-")
        fig.savefig(FIGURES / f"cm_{safe}.png", dpi=150)

    if show:
        plt.show()
    plt.close(fig)
    return metrics

#This function saves the results of one model as a new row in a CSV file
def save_results(metrics, csv_path, **extra):
    row = {**metrics, **{k: str(v) for k, v in extra.items()}}
    df = pd.DataFrame([row])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)