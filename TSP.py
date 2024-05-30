import numpy as np
import matplotlib.pyplot as plt
import heapq

# Generate random coordinates for cities
def generate_cities(num_cities):
    return np.random.rand(num_cities, 2)

# Calculate the cost matrix (Euclidean distance)
def calculate_cost_matrix(cities):
    num_cities = len(cities)
    cost_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                cost_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
    return cost_matrix

class Node:
    def __init__(self, level, path, bound):
        self.level = level  # Level in the state space tree
        self.path = path    # Current path
        self.bound = bound  # Lower bound of the path
    
    def __lt__(self, other):
        return self.bound < other.bound  # Comparison for priority queue

def calculate_bound(node, cost_matrix):
    bound = 0
    n = len(cost_matrix)
    for i in range(n):
        min_cost = float('inf')
        if i not in node.path:
            for j in range(n):
                if i != j and j not in node.path:
                    if cost_matrix[i][j] < min_cost:
                        min_cost = cost_matrix[i][j]
            bound += min_cost
    return bound

def branch_and_bound_tsp(cost_matrix):
    n = len(cost_matrix)
    pq = []
    initial_path = [0]  # Start from the first city
    initial_bound = calculate_bound(Node(0, initial_path, 0), cost_matrix)
    heapq.heappush(pq, Node(0, initial_path, initial_bound))
    best_cost = float('inf')
    best_path = []

    while pq:
        current_node = heapq.heappop(pq)
        if current_node.bound < best_cost:
            for i in range(1, n):
                if i not in current_node.path:
                    new_path = current_node.path + [i]
                    if len(new_path) == n:
                        new_path.append(0)  # Return to the starting city
                        cost = sum(cost_matrix[new_path[j]][new_path[j + 1]] for j in range(n))
                        if cost < best_cost:
                            best_cost = cost
                            best_path = new_path
                    else:
                        new_bound = calculate_bound(Node(current_node.level + 1, new_path, 0), cost_matrix)
                        if new_bound < best_cost:
                            heapq.heappush(pq, Node(current_node.level + 1, new_path, new_bound))

    return best_path, best_cost

# Plot the cities and the best path
def plot_tsp(cities, path):
    plt.figure(figsize=(10, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c='blue')
    for i in range(len(cities)):
        plt.text(cities[i, 0], cities[i, 1], f'{i}', fontsize=12, ha='right')
    for i in range(len(path) - 1):
        start, end = path[i], path[i + 1]
        plt.plot([cities[start, 0], cities[end, 0]], [cities[start, 1], cities[end, 1]], 'ro-')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('TSP Solution using Branch and Bound')
    plt.show()

def main():
    num_cities = 10  # You can change the number of cities here
    cities = generate_cities(num_cities)
    cost_matrix = calculate_cost_matrix(cities)
    best_path, best_cost = branch_and_bound_tsp(cost_matrix)

    print(f"Best path: {best_path}")
    print(f"Cost: {best_cost}")

    plot_tsp(cities, best_path)

if __name__ == "__main__":
    main()
