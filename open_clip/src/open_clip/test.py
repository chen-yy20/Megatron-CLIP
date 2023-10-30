import torch

def convert_tuple(*items):
    tuple_without_none = []
    for item in items:
        if item is None:
            tuple_without_none.append(torch.ones(1, 1))
        else:
            assert isinstance(item, torch.Tensor), f'ERROR! Pipeline stage output must be tensors, current {item}'
            tuple_without_none.append(item)
    return tuple(tuple_without_none)

def convert_back(tup):
    tuple_with_none = []
    none_tag = torch.ones(1, 1)
    for item in tup:
        tuple_with_none.append(None if item.shape == none_tag.shape and item == none_tag else item)
    return tuple(tuple_with_none)

def func1():
    a = [1,2,3]
    b = (1,2,3,4)
    return a

def func2():
    a = [1,2,3]
    b = (1,2,3,4)
    return b

def func3():
    a = [1,2,3]
    b = (1,2,3,4)
    return tuple(b)

def func4(*items):

    return *items,1,2,3,4

# print(func1())
# print(func2())
# print(func3())
# print(func4(7,9,8))

import networkx as nx
import matplotlib.pyplot as plt
g = nx.DiGraph(nx.nx_pydot.read_dot('/home/zanzong/workspace/open_clip/ds_configs/layerinfo_ViT-16-L-tiny.dot'))
# plt.savefig("layerinfo_ViT-16-L-tiny.png")
print(g.edges.data)
print(g.nodes.data)
for n, nbrs in g.adj.items():
    print(n, nbrs)
    for nbr in nbrs.keys():
        print(f"Edge {nbr}->{n}, with attribute {nbrs[nbr].keys()}={nbrs[nbr]['map_out_in']}")
        
# rg = g.reverse(copy=True)
# for n, nbrs in rg.adj.items():
#     print(n, nbrs)
    
# dag = nx.DiGraph()
# dag.add_nodes_from([1,2,3])
# dag.add_edges_from([(1, 2), (1, 3)])
# attrs = {(1, 2): {"attr1": "0_1"}, (1, 3): {"attr2": "1_2"}}
# nx.set_edge_attributes(dag, attrs)
# for n, nbrs in dag.adj.items():
#     print(n, nbrs)
# nx.nx_pydot.write_dot(dag, "out.dot")
