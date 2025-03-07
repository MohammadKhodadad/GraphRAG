import networkx as nx
import json

class GraphManager:
    def __init__(self):
        self.graph = nx.Graph()
    
    def add_node(self, node, source, description="",text="", meta_description={'pubchem':'','wikipedia':''}):
        """Adds a node with a description from a specific source."""
        if node not in self.graph.nodes:
            self.graph.add_node(node, description={})
        for key,value in meta_description.items():
            if value:
                if key not in self.graph.nodes[node]["description"]:
                    self.graph.nodes[node]["description"][key]=[value]
        if description:
            if source not in self.graph.nodes[node]["description"]:
                self.graph.nodes[node]["description"][source] = []
            self.graph.nodes[node]["description"][source].append({'description': description, 'text':text})
        
    
    def add_edge(self, node1, node2, source, description="",text=""):
        """Adds an edge between two nodes with a description from a specific source as a list."""
        if not self.graph.has_edge(node1, node2):
            self.graph.add_edge(node1, node2, description={})
        
        if description:
            if source not in self.graph.edges[node1, node2]["description"]:
                self.graph.edges[node1, node2]["description"][source] = []
            self.graph.edges[node1, node2]["description"][source].append({'description': description, 'text':text})
    
    
    def get_node_description(self, node):
        """Returns the description dictionary of a node."""
        return self.graph.nodes[node].get("description", {})
    
    def get_edge_description(self, node1, node2):
        """Returns the description dictionary of an edge."""
        return self.graph.edges[node1, node2].get("description", {})
    
    def save_graph(self, filename):
        """Saves the graph to a JSON file."""
        data = {
            "nodes": {node: self.graph.nodes[node] for node in self.graph.nodes},
            "edges": [
                (n1, n2, self.graph.edges[n1, n2])
                for n1, n2 in self.graph.edges
            ],
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
    
    def load_graph(self, filename):
        """Loads the graph from a JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)
        
        self.graph.clear()
        for node, attributes in data["nodes"].items():
            self.graph.add_node(node, **attributes)
        
        for node1, node2, attributes in data["edges"]:
            self.graph.add_edge(node1, node2, **attributes)
    
    def display_graph(self):
        """Displays the graph structure."""
        print("Nodes:")
        for node, data in self.graph.nodes(data=True):
            print(f"  {node}: {data}")
        print("\nEdges:")
        for edge in self.graph.edges(data=True):
            print(f"  {edge}")

# Example usage
if __name__ == "__main__":
    gm = GraphManager()
    gm.add_node("A", "User1", "Start node")
    gm.add_node("B", "User2", "Intermediate node")
    gm.add_edge("A", "B", "User1", "First connection")
    gm.add_edge("A", "B", "User2", "Another perspective")
    
    print(gm.get_node_description("A"))  # Output: {'User1': 'Start node'}
    print(gm.get_edge_description("A", "B"))  # Output: {'User1': 'First connection', 'User2': 'Another perspective'}
    
    gm.save_graph("graph.json")
    gm.display_graph()
    
    # Load and verify
    gm2 = GraphManager()
    gm2.load_graph("graph.json")
    gm2.display_graph()