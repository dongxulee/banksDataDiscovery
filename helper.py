import numpy as np 
import pandas as pd
from collections import deque
import networkx as nx
from matplotlib import pyplot as plt

df = pd.read_csv('df.csv')
df["REPDTE"] = pd.to_datetime(df["REPDTE"], format="%Y-%m-%d").dt.date

codes1 = ["ASSET", "CHBAL", "CHBALNI", "CHBALI", "CHCIC", "CHUS","CHNUS", "CHFRB", "SC",
        "SCUS", "SCMUNI", "SCDOMO", "SCFORD", "SCEQNFT", "SCHA", "FREPO", "LNLSNET", "LNRE", "LNAG",
        "LNCI", "LNCON", "LNOTCI", "LNCOMRE", "TRADE", "BKPREM", "ORE", "INTAN", "AOA"]

codes2 = ["LIABEQ", "LIAB", "DEP", "DEPDOM", "DEPFOR", "FREPP", 
         "TRADEL", "OTHBRF", "SUBND", "ALLOTHL", "EQTOT", "EQPP",
         "EQCS", "EQSUR", "EQUPTOT"]

filenames = ["assetStructure.txt", "liabilityStructure.txt"]
names = []
for filename in filenames:
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            name = line
            names.append(name)
name2code = dict(zip(names, codes1 + codes2))
code2name = dict(zip(codes1 + codes2, names))

def banks(year, month, numOfBanks):
    # The largest 100 banks
    dates = df["REPDTE"].unique()
    for day in dates:
        if day.year == year and day.month == month:
            largestBanks = df[(df['REPDTE'] == day)].groupby("NAMEFULL").mean().sort_values(by="ASSET", ascending=False).head(numOfBanks).index.values.tolist()
            dd = df[(df["NAMEFULL"].isin(largestBanks)) & (df['REPDTE'] == day)][["REPDTE", "NAMEFULL"]+list(name2code.values())]
            dd = dd.groupby("NAMEFULL").mean().sort_values(by="ASSET", ascending=False) 
    return dd, largestBanks

def bankPlot(bank,type,dd):
    filename = type + "Structure.txt"
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
                value = dd.loc[bank][name2code[name]].max()


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
        percentage = "" if graph.in_degree(node) == 0 else (value / parent_value)
        return node, percentage


    tree = read_tree_structure(filename)
    graph = create_directed_graph(tree)
    node_labels = []
    for node in graph.nodes:
        parent_value = graph.nodes[node]["value"] if graph.in_degree(node) == 0 else graph.nodes[list(graph.predecessors(node))[0]]["value"]
        node_labels.append(format_label(node, graph.nodes[node]["value"], parent_value))

    return node_labels


# return banks data frame with the given bank names
def sdf(year, month, bankNames, numOfBanks = 100):
    dd, _ = banks(year, month, numOfBanks)
    d = dd.loc[bankNames]
    L1 = []
    L2 = []
    for bank in bankNames:
        l1 = bankPlot(bank,'asset', d)
        l2 = bankPlot(bank,'liability', d)
        L1.append(np.array(l1)[1:,1].astype(float))
        L2.append(np.array(l2)[1:,1].astype(float))
    L1 = np.array(L1)
    L2 = np.array(L2)
    
    X = np.zeros(d[codes1 + codes2].shape)
    X[:,0] = 0.5
    X[:,1] = 0.5
    X[:,2:] = np.concatenate((L1,L2),axis=1)
    # fill na values in numpy array with 0
    X = np.nan_to_num(X, copy=False)
    Code = [codes1[0]] + [codes2[0]] + codes1[1:] + codes2[1:]
    names = [code2name[code] for code in Code]
    sdf = pd.DataFrame(X,columns=names)
    return sdf


def pairwiseComparison(years, months, numberOfBanks, xlabel = True):
    _, banks1 = banks(years[0], months[0], numberOfBanks)
    _, banks2 = banks(years[1], months[1], numberOfBanks)
    commonBanks = [bank for bank in banks1 if bank in banks2]
    
    index = np.arange(len(commonBanks))
    
    d1 = sdf(years[0], months[0], commonBanks)['Held to Maturity (Book Value)']
    d2 = sdf(years[1], months[1], commonBanks)['Held to Maturity (Book Value)']  
    if numberOfBanks > 10:
        diff = d2 - d1
        colors = ['blue' if val >= 0 else 'red' for val in diff]
        plt.bar(index, d2 - d1, color = colors)
        plt.ylabel("Change in HTM Securities Ratios")
    else:
        width = 0.30
        plt.bar(index, d1, width, label = "2022")
        plt.bar(index+width, d2, width, label = "2023")
        plt.ylabel("HTM Securities Ratios")
    plt.xlabel("Banks ID")
    if xlabel:
        plt.xticks(range(len(commonBanks)), commonBanks, rotation='vertical')
        plt.xlabel("Banks")
        plt.legend()
    return d1,d2