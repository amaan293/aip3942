from mnist_preprocess import load_mnist, preprocess, stratified_split
from svm_mnist import (
    train_linear_svm, predict_linear_svm,
    train_quadratic_svm, predict_quadratic_svm,
    accuracy, precision, recall, f1_score
)
from tune_visualise import tune_linear_svm, tune_quadratic_svm, plot_decision_boundary_2d
import matplotlib.pyplot as plt

def evaluate_and_print(name, X, y, predict_fn, params):
    y_pred = predict_fn(X, *params)
    print(f"{name} : Acc={accuracy(y, y_pred):.4f}, "
          f"Prec={precision(y, y_pred):.4f}, "
          f"Rec={recall(y, y_pred):.4f}, "
          f"F1={f1_score(y, y_pred):.4f}")
    return {
        'Accuracy': accuracy(y, y_pred),
        'Precision': precision(y, y_pred),
        'Recall': recall(y, y_pred),
        'F1': f1_score(y, y_pred)
    }

def plot_metrics_comparison(metrics_linear, metrics_quad, title="Linear vs Quadratic SVM (1k samples)"):
    labels = list(metrics_linear.keys())
    linear_vals = list(metrics_linear.values())
    quad_vals = list(metrics_quad.values())
    x = range(len(labels))
    plt.figure(figsize=(8,5))
    plt.bar([i-0.15 for i in x], linear_vals, width=0.3, label="Linear 1k")
    plt.bar([i+0.15 for i in x], quad_vals, width=0.3, label="Quadratic 1k")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0,1)
    plt.legend()
    plt.show()

def main():
    data_dir = r"D:\aip3942\W-7\mnist_dataset"
    X_train_img, y_train, X_test_img, y_test = load_mnist(data_dir)
    X_train, X_test = preprocess(X_train_img), preprocess(X_test_img)
    X_tr, y_tr, X_val, y_val = stratified_split(X_train, y_train, val_ratio=0.1)

    print("Full dataset shapes:", X_tr.shape, X_val.shape, X_test.shape)

    n_lin_small = 1000
    X_lin_small, y_lin_small = X_tr[:n_lin_small], y_tr[:n_lin_small]
    print(f"\nRunning Linear SVM on {n_lin_small} samples...")
    W_lin_small, b_lin_small, classes_lin_small = train_linear_svm(
        X_lin_small, y_lin_small, C=1.0, lr=1e-3, epochs=5
    )
    metrics_linear_1k_train = evaluate_and_print("Train (Linear 1k)", X_lin_small, y_lin_small,
                                                 predict_linear_svm, (W_lin_small, b_lin_small, classes_lin_small))
    metrics_linear_1k_val = evaluate_and_print("Val   (Linear 1k)", X_val, y_val,
                                               predict_linear_svm, (W_lin_small, b_lin_small, classes_lin_small))
    metrics_linear_1k_test = evaluate_and_print("Test  (Linear 1k)", X_test, y_test,
                                                predict_linear_svm, (W_lin_small, b_lin_small, classes_lin_small))

    n_quad_train = 1000
    X_quad_train, y_quad_train = X_tr[:n_quad_train], y_tr[:n_quad_train]
    print(f"\nRunning Quadratic SVM on {n_quad_train} samples...")
    alphas, b_quad, classes_quad, K_train, degree, X_tr_quad, y_tr_quad = train_quadratic_svm(
        X_quad_train, y_quad_train, C=1.0, lr=1e-4, epochs=20, degree=2
    )
    metrics_quad_1k_train = evaluate_and_print("Train (Quadratic 1k)", X_quad_train, y_quad_train,
                                               predict_quadratic_svm, (alphas, b_quad, classes_quad, degree, X_tr_quad, y_tr_quad))
    metrics_quad_1k_val = evaluate_and_print("Val   (Quadratic 1k)", X_val, y_val,
                                             predict_quadratic_svm, (alphas, b_quad, classes_quad, degree, X_tr_quad, y_tr_quad))
    metrics_quad_1k_test = evaluate_and_print("Test  (Quadratic 1k)", X_test, y_test,
                                              predict_quadratic_svm, (alphas, b_quad, classes_quad, degree, X_tr_quad, y_tr_quad))

    print(f"\nRunning Linear SVM on full training set ({len(X_tr)} samples)...")
    W_lin_full, b_lin_full, classes_lin_full = train_linear_svm(
        X_tr, y_tr, C=1.0, lr=1e-3, epochs=5
    )
    evaluate_and_print("Train (Linear Full)", X_tr, y_tr,
                       predict_linear_svm, (W_lin_full, b_lin_full, classes_lin_full))
    evaluate_and_print("Val   (Linear Full)", X_val, y_val,
                       predict_linear_svm, (W_lin_full, b_lin_full, classes_lin_full))
    evaluate_and_print("Test  (Linear Full)", X_test, y_test,
                       predict_linear_svm, (W_lin_full, b_lin_full, classes_lin_full))

    C_values = [0.00001, 0.01, 0.1, 1.0]
    degree_values = [2]

    print("\nTuning Linear SVM...")
    tune_linear_svm(X_tr, y_tr, X_val, y_val, C_values)

    print("\nTuning Quadratic SVM...")
    tune_quadratic_svm(X_quad_train, y_quad_train, X_val, y_val, C_values, degree_values)

    print("\nPlotting Linear decision boundary (1k)...")
    plot_decision_boundary_2d(X_lin_small, y_lin_small, predict_linear_svm,
                              (W_lin_small, b_lin_small, classes_lin_small),
                              "Linear SVM Decision Boundary (2D PCA, 1k samples)")

    print("\nPlotting Quadratic decision boundary (1k)...")
    plot_decision_boundary_2d(X_quad_train, y_quad_train, predict_quadratic_svm,
                              (alphas, b_quad, classes_quad, degree, X_tr_quad, y_tr_quad),
                              "Quadratic SVM Decision Boundary (2D PCA, 1k samples)")

    print("\nPlotting Linear 1k vs Quadratic 1k metrics...")
    plot_metrics_comparison(metrics_linear_1k_test, metrics_quad_1k_test)

    print("\nEvaluating Linear SVM with different C values: ")
    metrics_vs_C_lin = {C: {} for C in C_values}
    for C in C_values:
        W, b, classes = train_linear_svm(X_tr, y_tr, C=C, lr=1e-3, epochs=5)
        y_val_pred = predict_linear_svm(X_val, W, b, classes)
        metrics_vs_C_lin[C] = {
            'Accuracy': accuracy(y_val, y_val_pred),
            'Precision': precision(y_val, y_val_pred),
            'Recall': recall(y_val, y_val_pred),
            'F1': f1_score(y_val, y_val_pred)
        }
        print(f"Linear C={C}: {metrics_vs_C_lin[C]}")
    plt.figure(figsize=(8,6))
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
        plt.plot(C_values, [metrics_vs_C_lin[C][metric] for C in C_values], marker='o', label=metric)
    plt.xscale("log")
    plt.xlabel("C (log scale)")
    plt.ylabel("Score")
    plt.title("Linear SVM Metrics vs C (Validation Set)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nEvaluating Quadratic SVM with different C values: ")
    metrics_vs_C_quad = {C: {} for C in C_values}
    for C in C_values:
        alphas, b_q, classes_q, _, degree, Xq, yq = train_quadratic_svm(
            X_quad_train, y_quad_train, C=C, lr=1e-4, epochs=20, degree=2
        )
        y_val_pred = predict_quadratic_svm(X_val, alphas, b_q, classes_q, degree, Xq, yq)
        metrics_vs_C_quad[C] = {
            'Accuracy': accuracy(y_val, y_val_pred),
            'Precision': precision(y_val, y_val_pred),
            'Recall': recall(y_val, y_val_pred),
            'F1': f1_score(y_val, y_val_pred)
        }
        print(f"Quadratic C={C}: {metrics_vs_C_quad[C]}")
    plt.figure(figsize=(8,6))
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
        plt.plot(C_values, [metrics_vs_C_quad[C][metric] for C in C_values], marker='o', label=metric)
    plt.xscale("log")
    plt.xlabel("C (log scale)")
    plt.ylabel("Score")
    plt.title("Quadratic SVM Metrics vs C (Validation Set)")
    plt.legend()
    plt.grid(True)
    plt.show()

main()
