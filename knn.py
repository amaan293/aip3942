import csv
import random
from collections import Counter

# Manhattan distance function
def manhattan_distance(row1, row2):
    return sum(abs(a - b) for a, b in zip(row1, row2))

# Load data from iris.csv
def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            features = list(map(float, row[1:5]))  # 4 features
            label = row[5]  # species
            dataset.append((features, label))
    return dataset

# Split into train and test sets
def train_test_split(data, test_ratio=0.2):
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_ratio))
    return data[:split_index], data[split_index:]

# k-NN implementation
def knn_predict(test_row, train_data, k=3):
    distances = []
    for train_row in train_data:
        dist = manhattan_distance(test_row, train_row[0])
        distances.append((dist, train_row[1]))
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    labels = [label for _, label in k_nearest]
    
    print(f"\nTest sample: {test_row}")
    print("Manhattan Distances from training points:")
    for d, l in k_nearest:
        print(f"Distance: {d:.2f}, Label: {l}")
        
    most_common = Counter(labels).most_common(1)[0][0]
    return most_common

# Main
def main():
    filename = 'iris.csv'  # Ensure this file is in the same directory
    data = load_dataset(filename)
    
    train_data, test_data = train_test_split(data, test_ratio=0.2)
    
    correct = 0
    for features, actual_label in test_data:
        predicted_label = knn_predict(features, train_data, k=3)
        print(f"Predicted: {predicted_label}, Actual: {actual_label}")
        if predicted_label == actual_label:
            correct += 1
            
    accuracy = correct / len(test_data) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
