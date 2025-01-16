from main import MODEL_CLASSES_AND_PARAMETERS, X_train, Y_train, X_test, Y_test
from plot_confusion_metrics import plot_confusion_matrix_and_print_metrics


def train_and_evaluate_model(model_name):
    model_class, model_params = MODEL_CLASSES_AND_PARAMETERS[model_name]
    model = model_class(**model_params)
    model.fit(X_train, Y_train)
    y_test_pred = model.predict(X_test)
    plot_confusion_matrix_and_print_metrics(y_test_pred, Y_test, model_params, model_name)
    print("\n"*5)
