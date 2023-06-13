import re
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from collections import deque
import pandas as pd 

date = '2023-03-31'

banks = ["BANK OF AMERICA, NATIONAL ASSOCIATION"
]



df = pd.read_csv('df.csv')

codes1 = ["ASSET", "CHBAL", "CHBALNI", "CHBALI", "CHCIC", "CHUS","CHNUS", "CHFRB", "SC",
        "SCUS", "SCMUNI", "SCDOMO", "SCFORD", "SCEQNFT", "SCHA", "FREPO", "LNLSNET", "LNRE", "LNAG",
        "LNCI", "LNCON", "LNOTCI", "LNCOMRE", "TRADE", "BKPREM", "ORE", "INTAN", "AOA"]


codes2 = ["LIABEQ", "LIAB", "DEP", "DEPDOM", "DEPFOR", "FREPP", 
         "TRADEL", "OTHBRF", "SUBND", "ALLOTHL", "EQTOT", "EQPP",
         "EQCS", "EQSUR", "EQUPTOT"]

def bankPlot(bank, type):
    if type == "asset":
        codes = codes1
        filename = "assetStructure.txt"
    else:
        codes = codes2
        filename = "liabilityStructure.txt"
    names = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            name = line
            names.append(name)
    name2code = dict(zip(names, codes))
        
    def read_tree_structure(filename):
        def count_indentation(line):
            count = 0
            for char in line:
                if char == ' ':
                    count += 1
                elif char == '\t':
                    count += 4
                else:
                    break
            return count

        tree = {}
        with open(filename) as f:
            lines = f.readlines()
            stack = deque()
            for line in lines:
                depth = count_indentation(line) // 4
                line = line.strip()
                name = line
                value = df[(df['REPDTE'] == date) & (df["NAMEFULL"] == bank)][name2code[name]].max()


                node = {"name": name, "children": [], "depth": depth, "value": value}

                # Pop elements from the stack until reaching the correct depth
                while stack and stack[-1]["depth"] >= depth:
                    stack.pop()

                # If the stack is empty, the node is the root
                if not stack:
                    tree = node
                else:
                    # Otherwise, append the node to the parent's children
                    stack[-1]["children"].append(node)

                stack.append(node)
        return tree


    def create_directed_graph(tree):
        graph = nx.DiGraph()
        def traverse(node):
            graph.add_node(node["name"], value=node["value"], depth=node["depth"])
            for child in node["children"]:
                graph.add_edge(node["name"], child["name"])
                traverse(child)
        traverse(tree)
        return graph


    def format_label(node, value, parent_value):
        percentage = "" if graph.in_degree(node) == 0 else f" ({value / parent_value:.0%})"
        return f"{node}{percentage}"


    def custom_node_positions(graph):
        node_depths = nx.get_node_attributes(graph, 'depth')
        positions = {}
        level_counts = {}
        depth_count = {}
        for node, depth in node_depths.items():
            depth_count[depth] = depth_count.get(depth, 0) + 1
            level_counts[depth] = level_counts.get(depth, 0) + 1
            positions[node] = (depth, level_counts[depth])
        normalized_positions = {
            node: (
                max(depth / max(node_depths.values()) - 0.3, 0), 
                y / depth_count[depth]
            ) for node, (depth, y) in positions.items()
        }

        return normalized_positions

    tree = read_tree_structure(filename)
    graph = create_directed_graph(tree)
    node_positions = custom_node_positions(graph)
    node_positions

    node_labels = list(graph.nodes)
    link_sources = [node_labels.index(link[0]) for link in graph.edges]
    link_targets = [node_labels.index(link[1]) for link in graph.edges]
    link_values = [graph.nodes[link[1]]["value"] for link in graph.edges]

    node_labels = []
    for node in graph.nodes:
        parent_value = graph.nodes[node]["value"] if graph.in_degree(node) == 0 else graph.nodes[list(graph.predecessors(node))[0]]["value"]
        node_labels.append(format_label(node, graph.nodes[node]["value"], parent_value))

    sankey_nodes = dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=node_labels,
        x=[pos[0] for pos in node_positions.values()],
        y=[pos[1] for pos in node_positions.values()]
    )

    sankey_fig = go.Figure(data=[go.Sankey(
        node=sankey_nodes,
        link=dict(source=link_sources, target=link_targets, value=link_values)
    )])

    sankey_fig.update_layout(title_text="Balance Sheet Sankey Diagram: " + bank +" at "+ date, font_size=10, height=800, width=1000)
    sankey_fig.show()

for bank in banks:
    bankPlot(bank, "asset")