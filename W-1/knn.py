import csv
import random

def manhattan_distance(row1, row2):
    total = 0
    for i in range(len(row1)):
        diff = row1[i] - row2[i]
        if diff < 0:
            diff = -diff
        total += diff
    return total

def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            features = list(map(float, row[1:5]))  # 4 cloumns are features
            label = row[5] #last column is label
            dataset.append((features, label))
    return dataset

def train_test_split(data, test_ratio=0.2):
    # Shuffle first to remove label order bias
    for i in range(len(data) - 1, 0, -1):
        j = random.randint(0, i)
        data[i], data[j] = data[j], data[i]
    split_index = int(len(data) * (1 - test_ratio))
    return data[:split_index], data[split_index:]

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j][0] > arr[j + 1][0]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def sort_dict_items_by_value_desc(d):
    items = list(d.items())
    n = len(items)
    for i in range(n):
        for j in range(0, n - i - 1):
            if items[j][1] < items[j + 1][1]:
                items[j], items[j + 1] = items[j + 1], items[j]
    return items

def knn_predict(test_row, train_data, k=5):
    distances = []
    for train_row in train_data:
        dist = manhattan_distance(test_row, train_row[0])
        distances.append((dist, train_row[1]))

    distances = bubble_sort(distances)
    k_nearest = distances[:k]
    labels = [label for _, label in k_nearest]

    label_counts = {}
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    print("\nLabel counts among k nearest neighbors:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    sorted_labels = sort_dict_items_by_value_desc(label_counts)
    most_common = sorted_labels[0][0]

    print(f"→ Predicted label based on majority vote: {most_common}")
    print(f"\nTest sample: {test_row}")
    print("Manhattan Distances from training points:")
    for d, l in k_nearest:
        print(f"Distance: {d:.2f}, Label: {l}")

    return most_common

def main():
    filename = 'iris.csv'
    data = load_dataset(filename)

    train_data, test_data = train_test_split(data, test_ratio=0.2)

    correct = 0
    total = len(test_data)

    for features, actual_label in test_data:
        predicted_label = knn_predict(features, train_data, k=5)
        print(f"Predicted: {predicted_label}, Actual: {actual_label}")
        if predicted_label == actual_label:
            correct += 1

    accuracy = (correct / total) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
