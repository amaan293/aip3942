import numpy as np
import matplotlib.pyplot as plt
import csv
import random

# Load CSV manually
def load_csv(filename):
    data = []
    with open(filename, 'r') as file:
        lines = csv.reader(file)
        next(lines)  # column names reomoval
        for row in lines:
            data.append([float(val) for val in row])
    return np.array(data)

# Manual train-test split
def manual_split(X, y, test_ratio):
    indices = list(range(len(X)))
    random.seed(42)
    random.shuffle(indices)
    split = int(len(X) * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

# Accuracy
def get_accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)

def manhattan_distance(row1, row2):
    total = 0
    for i in range(len(row1)):
        diff = row1[i] - row2[i]
        if diff < 0:
            diff = -diff
        total += diff
    return total

# k-NN from scratch
def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = []
        for i in range(len(X_train)):
            dist = manhattan_distance(test_point, X_train[i])
            distances.append((dist, y_train[i]))
        distances.sort()
        neighbors = [label for _, label in distances[:k]]
        counts = {}
        for label in neighbors:
            counts[label] = counts.get(label, 0) + 1
        predicted = max(counts, key=counts.get)
        predictions.append(predicted)
    return np.array(predictions)

# Naive Bayes from scratch
def naive_bayes_train(X_train, y_train):
    classes = np.unique(y_train)
    summaries = {}
    priors = {}

    for cls in classes:
        X_cls = X_train[y_train == cls]
        mean = np.mean(X_cls, axis=0)
        var = np.var(X_cls, axis=0) + 1e-6  
        summaries[cls] = (mean, var)
        priors[cls] = len(X_cls) / len(X_train)
    return summaries, priors

def gaussian_prob(x, mean, var):
    return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)

def naive_bayes_predict(X_test, summaries, priors):
    predictions = []
    for x in X_test:
        posteriors = {}
        for cls in summaries:
            mean, var = summaries[cls]
            probs = gaussian_prob(x, mean, var)
            log_prob = np.sum(np.log(probs))
            log_prior = np.log(priors[cls])
            posteriors[cls] = log_prior + log_prob
        predicted = max(posteriors, key=posteriors.get)
        predictions.append(predicted)
    return np.array(predictions)

# Main
if __name__ == "__main__":
    data = load_csv('W-2/diabetes.csv')
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    split_ratios = [0.3, 0.2]  
    k_values = [3, 5, 7, 9,11,13,15,17]

    for ratio in split_ratios:
        print(f"\n Train:Test Split = {int((1 - ratio)*100)}:{int(ratio*100)}")
        knn_accuracies = []
        X_train, y_train, X_test, y_test = manual_split(X, y, ratio)

        # Naive Bayes
        summaries, priors = naive_bayes_train(X_train, y_train)
        nb_pred = naive_bayes_predict(X_test, summaries, priors)
        nb_acc = get_accuracy(y_test, nb_pred)
        print(f"Naive Bayes Accuracy: {nb_acc*100:.2f}%")

        # k-NN
        for k in k_values:
            knn_pred = knn_predict(X_train, y_train, X_test, k)
            acc = get_accuracy(y_test, knn_pred)
            knn_accuracies.append(acc)
            print(f"k-NN Accuracy (k={k}): {acc*100:.2f}%")

        # Plot
        plt.figure()
        plt.plot(k_values, knn_accuracies, marker='o', label='k-NN')
        plt.plot(k_values, [nb_acc]*len(k_values), linestyle='--', marker='x', label='Naive Bayes')
        plt.title(f'Accuracy for {int((1 - ratio)*100)}:{int(ratio*100)} Train:Test Split')
        plt.xlabel('k value')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()
