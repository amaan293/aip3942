"""
 def main():
    data_dir = r"D:\aip3942\W-7\mnist_dataset"

    X_train_img, y_train, X_test_img, y_test = load_mnist(data_dir)
    X_train, X_test = preprocess(X_train_img), preprocess(X_test_img)

    X_tr, y_tr, X_val, y_val = stratified_split(X_train, y_train)

    print("Training set:", X_tr.shape, y_tr.shape)
    print("Validation set:", X_val.shape, y_val.shape)
    print("Test set:", X_test.shape, y_test.shape)

if __name__ == "__main__":
    main()


THIS WAS FOR THE RUNNING A MAIN AND CHECKING FOR PREPROCESSING
    """
    
    
"""def main():
    data_dir = r"D:\aip3942\W-7\mnist_dataset"

    X_train_img, y_train, X_test_img, y_test = load_mnist(data_dir)
    X_train, X_test = preprocess(X_train_img), preprocess(X_test_img)
    X_tr, y_tr, X_val, y_val = stratified_split(X_train, y_train)

    n_train, n_val, n_test = 500, 200, 200
    X_tr, y_tr = X_tr[:n_train], y_tr[:n_train]
    X_val, y_val = X_val[:n_val], y_val[:n_val]
    X_test, y_test = X_test[:n_test], y_test[:n_test]

    print("Training shapes:", X_tr.shape, y_tr.shape)
    print("Validation shapes:", X_val.shape, y_val.shape)
    print("Test shapes:", X_test.shape, y_test.shape)

    W, b_lin, classes_lin = train_linear_svm(X_tr, y_tr, C=1.0, lr=1e-3, epochs=5)
    for name, X, y in [("Train", X_tr, y_tr),
                       ("Val", X_val, y_val),
                       ("Test", X_test, y_test)]:
        y_pred = predict_linear_svm(X, W, b_lin, classes_lin)
        print(f"\nLinear SVM {name} Results:")
        print(f"  Accuracy : {accuracy(y, y_pred):.4f}")
        print(f"  Precision: {precision(y, y_pred):.4f}")
        print(f"  Recall   : {recall(y, y_pred):.4f}")
        print(f"  F1-score : {f1_score(y, y_pred):.4f}")

    alphas, b_quad, classes_quad, K_train, degree, X_tr_quad, y_tr_quad = train_quadratic_svm(
        X_tr, y_tr, C=1.0, lr=1e-4, epochs=20, degree=2
    )
    for name, X, y in [("Train", X_tr, y_tr),
                       ("Val", X_val, y_val),
                       ("Test", X_test, y_test)]:
        y_pred = predict_quadratic_svm(X, alphas, b_quad, classes_quad, degree, X_tr_quad, y_tr_quad)
        print(f"\nQuadratic SVM {name} Results:")
        print(f"  Accuracy : {accuracy(y, y_pred):.4f}")
        print(f"  Precision: {precision(y, y_pred):.4f}")
        print(f"  Recall   : {recall(y, y_pred):.4f}")
        print(f"  F1-score : {f1_score(y, y_pred):.4f}")

if __name__ == "__main__":
    main()

THIS WAS A MAIN THAT RAN SVM_MNIST.PY
    """



"""def main():
    data_dir = r"D:\aip3942\W-7\mnist_dataset"
    X_train_img, y_train, X_test_img, y_test = load_mnist(data_dir)
    X_train, X_test = preprocess(X_train_img), preprocess(X_test_img)
    X_tr, y_tr, X_val, y_val = stratified_split(X_train, y_train)

    n_train, n_val, n_test = 500, 200, 200
    X_tr, y_tr = X_tr[:n_train], y_tr[:n_train]
    X_val, y_val = X_val[:n_val], y_val[:n_val]
    X_test, y_test = X_test[:n_test], y_test[:n_test]

    C_values = [0.01, 0.1, 1.0, 10.0]
    degree_values = [2, 3]

    linear_results = tune_linear_svm(X_tr, y_tr, X_val, y_val, C_values)
    quadratic_results = tune_quadratic_svm(X_tr, y_tr, X_val, y_val, C_values, degree_values)

    W, b, classes = train_linear_svm(X_tr, y_tr, C=1.0, lr=1e-3, epochs=5)
    plot_decision_boundary_2d(X_tr, y_tr, predict_linear_svm, (W, b, classes), "Linear SVM Decision Boundary (2D PCA)")

    alphas, b_quad, classes_quad, K_train, degree, X_tr_quad, y_tr_quad = train_quadratic_svm(
        X_tr, y_tr, C=1.0, lr=1e-4, epochs=20, degree=2
    )
    plot_decision_boundary_2d(X_tr, y_tr, predict_quadratic_svm,
                              (alphas, b_quad, classes_quad, degree, X_tr_quad, y_tr_quad),
                              "Quadratic SVM Decision Boundary (2D PCA)")

if __name__ == "__main__":
    main()
    
    
    THIS WAS A MAIN THAT RAN TUNE_VISUALISE.PY
    """