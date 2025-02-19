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

    def sample_random_paths(self, length, num_samples=5):
        """Randomly samples paths of the given length using streaming BFS instead of loading all paths."""
        sampled_paths = []
        nodes = list(self.graph_manager.graph.nodes)

        if not nodes:
            print("Graph is empty. No nodes to explore.")
            return []

        while len(sampled_paths) < num_samples:
            start = random.choice(nodes)  # Pick a random starting node
            queue = deque([(start, [start])])  # (current node, path taken)

            while queue:
                node, path = queue.popleft()

                # If the exact length is reached, consider it for sampling
                if len(path) == length + 1:
                    sampled_paths.append(path)
                    if len(sampled_paths) >= num_samples:
                        break  # Stop when enough samples are collected
                    continue  # Do not expand further

                # Expand neighbors while avoiding cycles
                for neighbor in self.graph_manager.graph.neighbors(node):
                    if neighbor not in path:  # Ensures simple paths (no cycles)
                        queue.append((neighbor, path + [neighbor]))

        if not sampled_paths:
            print(f"No paths of length {length} found.")
            return []

        return self._format_paths(sampled_paths)

    def _format_paths(self, sampled_paths):
        """Formats paths by retrieving edge descriptions."""
        formatted_paths = []
        for path in sampled_paths:
            completed_path = []
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                edge_data = self.graph_manager.graph.edges.get((node1, node2), {})
                description = edge_data.get("description", {})

                if description:
                    for key, value in description.items():
                        relation = value[-1].get('description', 'Unknown')
                        text = value[-1].get('text', '')
                        completed_path.append((node1, relation, node2, text))
            
            formatted_paths.append(completed_path)
        return formatted_paths

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
    print(explorer.sample_random_paths(2))
