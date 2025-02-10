import tqdm
import json
import random
import networkx as nx
from collections import deque
if __name__ == "__main__":
    from graph_manager import GraphManager
else:
    from .graph_manager import GraphManager

class GraphExplorer:
    def __init__(self, graph_manager):
        """Initializes the GraphExplorer with an instance of GraphManager."""
        self.graph_manager = graph_manager

    def find_all_paths_of_length_n(self, length):
        """Finds all paths of exact length `length` from all nodes using BFS."""
        all_paths = []

        for start in tqdm.tqdm(self.graph_manager.graph.nodes):
            queue = deque([(start, [start])])  # (current node, path taken)

            while queue:
                node, path = queue.popleft()

                # If we have reached the exact required length, add to results
                if len(path) == length + 1:
                    all_paths.append(path)
                    continue  # Don't expand this path further

                # Expand neighbors while avoiding cycles
                for neighbor in self.graph_manager.graph.neighbors(node):
                    if neighbor not in path:  # Ensures simple paths
                        queue.append((neighbor, path + [neighbor]))

        return all_paths

    def sample_random_paths(self, length, num_samples=5):
        """Randomly samples paths of the given length from all possible paths."""
        paths = self.find_all_paths_of_length_n(length)
        if not paths:
            print(f"No paths of length {length} found.")
            return []
        
        sampled_paths = random.sample(paths, min(num_samples, len(paths)))
        completed_sampled_paths=[]
        
        for path in sampled_paths:
            completed_path=[]
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                description = self.graph_manager.graph.edges[node1, node2].get("description")
                print(description)
                if len(description)>0:
                    for key,value in description.items():
                        relation = value.get('description')
                        text = value.get('text')
                else:
                    continue
                completed_path.append((node1, relation, node2, text))
            completed_sampled_paths.append(completed_path)
        return completed_sampled_paths

    def display_path(self, path):

        path_str = []
        
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            print('\nNODE:')
            print(f"{node1}: ",self.graph_manager.graph.nodes[node1].get("description", "No description"))
            print('\nEDGE:')
            print(self.graph_manager.graph.edges[node1, node2].get("description", "No description"))
            path_str.append(f"{node1} ->")
        path_str.append(f"{node2}")
        print('\nNODE:')
        print(f"{node2}: ",self.graph_manager.graph.nodes[node2].get("description", "No description"))
        print(" ".join(path_str))

# Example usage
if __name__ == "__main__":
    gm = GraphManager()
    gm.add_node("A", "User1", "Start node")
    gm.add_node("B", "User2", "Intermediate node")
    gm.add_node("C", "User3", "Another node")
    gm.add_node("D", "User4", "Extra node")
    gm.add_edge("A", "B", "User1", "First connection")
    gm.add_edge("B", "C", "User2", "Second connection")
    gm.add_edge("C", "D", "User3", "Third connection")
    gm.add_edge("A", "D", "User4", "Alternate connection")

    gm.display_graph()

    explorer = GraphExplorer(gm)

    path_length = 2
    num_samples = 3
    explorer.display_random_paths(path_length, num_samples)
