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

    def sample_random_paths(self, length, num_samples=5, min_chars=4):
        """Randomly samples paths of the given length using streaming BFS instead of loading all paths."""
        sampled_paths = []
        # nodes = list(self.graph_manager.graph.nodes)
        nodes = [node for node in self.graph_manager.graph.nodes if len(str(node)) >= min_chars]
        if not nodes:
            print("Graph is empty. No nodes to explore.")
            return []

        while len(sampled_paths) < num_samples:
            start = random.choice(nodes)  # Pick a random starting node
            queue = deque([(start, [start])])  # (current node, path taken)
            while queue:
                node, path = queue.popleft()

                # If the exact length is reached, format and store the path
                if len(path) == length + 1:
                    formatted_path, num_sources = self._format_path(path)
                    if formatted_path and (num_sources>1 or length==1):
                        sampled_paths.append(formatted_path)
                        break
                    continue  # Do not expand further

                # Expand neighbors while avoiding cycles

                neighbors = list(self.graph_manager.graph.neighbors(node))
                random.shuffle(neighbors)  # Shuffle the neighbors to randomize selection
                for neighbor in neighbors:
                    if neighbor not in path and len(str(neighbor)) >= min_chars:
                        queue.append((neighbor, path + [neighbor]))

        if not sampled_paths:
            print(f"No paths of length {length} found.")
            return []

        return sampled_paths

    def _format_path(self, path):
        """Formats a single path by retrieving edge descriptions and sources, ensuring diverse sources are picked."""
        completed_path = []
        used_sources = set()
        num_sources = 0

        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            edge_data = self.graph_manager.graph.edges.get((node1, node2), {})
            descriptions = edge_data.get("description", {})
            available_sources = list(descriptions.keys())
            meta1=''
            pubchem_info=self.graph_manager.graph.nodes[node1].get('description',{}).get('pubchem','')
            if pubchem_info:
                meta1+=f'pubchem: {pubchem_info}\n'
            wiki_info=self.graph_manager.graph.nodes[node1].get('description',{}).get('wikipedia','')
            if wiki_info:
                meta1+=f'wikipedia: {wiki_info}\n'
            
            if available_sources:
                # Try to pick a new source each time
                new_source = next((s for s in available_sources if s not in used_sources), available_sources[0])
                used_sources.add(new_source)
                num_sources += 1
                value = descriptions[new_source]
                
                if value:
                    relation = value[-1].get('description', 'Unknown')
                    text = value[-1].get('text', '')
                    completed_path.append((node1, relation, node2, text, new_source, meta1))
        
        return completed_path, num_sources

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
    gm.add_node("AA", "User1", "Start node")
    gm.add_node("B", "User2", "Intermediate node")
    gm.add_node("BB", "User2", "Intermediate node")
    gm.add_node("CC", "User3", "Another node")
    gm.add_node("C", "User3", "Another node")
    gm.add_node("D", "User4", "Extra node")
    gm.add_edge("AA", "BB", "User1", "First connection")
    # gm.add_edge("BB", "CC", "User1.2", "Second connection")
    gm.add_edge("BB", "C", "User2", "Second.one connection")
    gm.add_edge("C", "D", "User1", "Third connection")
    gm.add_edge("CC", "DD", "User1.3", "Third.one connection")

    gm.display_graph()
    print()

    explorer = GraphExplorer(gm)
    path_length = 2
    num_samples = 1
    results = explorer.sample_random_paths(path_length, num_samples,min_chars=2)
    

    print(results)

    explorer = GraphExplorer(gm)
    path_length = 1
    num_samples = 1
    results = explorer.sample_random_paths(path_length, num_samples)
    

    print(results)
