import pandas as pd
import networkx as nx

class BiomedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_dataframe(self, df, source_col, relation_col, target_col):
        for _, row in df.iterrows():
            s = str(row[source_col]).strip()
            r = str(row[relation_col]).strip()
            t = str(row[target_col]).strip()
            self.graph.add_node(s)
            self.graph.add_node(t)
            self.graph.add_edge(s, t, relation=r)

    def statistics(self):
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph)
        }

    def save_graphml(self, path):
        nx.write_graphml(self.graph, path)
