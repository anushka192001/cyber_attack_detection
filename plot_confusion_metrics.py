from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# helper function to automate model training and evaluation processes
def plot_confusion_matrix_and_print_metrics(y_pred, y_true, model_params, model_name=""):
    conf_matrix = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(conf_matrix , annot=True, annot_kws={"size": 16}, fmt="d")
    if model_name:
        ax.set_title(f"Evaluation Results for {model_name} model with parameters {model_params}", fontsize=14, pad=20)
    ax.set_xlabel("Predicted Values", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(['Benign', 'Malicious'])
    ax.set_ylabel("Actual Values", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(['Benign', 'Malicious'])
    plt.show()