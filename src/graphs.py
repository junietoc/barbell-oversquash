import networkx as nx
import torch
from torch_geometric.utils import to_networkx, from_networkx



def barbell_data(n=200, clique_size=30, to_device="cpu"):
    """
    Crea un grafo 'barbell': dos cliques de tamaño `clique_size`
    unidas por una cadena lineal de (n - 2*clique_size) nodos.
    """
    assert n > 2 * clique_size, "n debe ser mayor que 2*clique_size"
    g = nx.barbell_graph(clique_size, n - 2*clique_size)
    data = from_networkx(g)           # Data(edge_index, num_nodes, …)
    # Features: 1‑hot del nodo (diagnóstico simple)
    data.x = torch.eye(data.num_nodes)
    data.y = torch.zeros(data.num_nodes, dtype=torch.long)
    data.y[:clique_size] = 0          # componente izquierda
    data.y[-clique_size:] = 1         # componente derecha
    return data.to(to_device)

def add_edges_balanced_forman(data, k=2):
    """
    Re‑cablea el grafo PyG 'data' añadiendo k aristas
    según curvatura Forman simplificada.  Devuelve un
    NUEVO objeto Data con las MISMAS x, y, etc.
    """
    device = data.x.device
    g = to_networkx(data, to_undirected=True)

    # --- Forman curvature (sin pesos) ---
    curv = {e: 4 - g.degree[e[0]] - g.degree[e[1]] for e in g.edges()}
    worst = sorted(curv, key=curv.get)[:k]

    # añadir una arista "contrapuesta" por cada e con curv. mínima
    n = g.number_of_nodes()
    for u, v in worst:
        for w in (u, v):
            target = (w + n // 2) % n
            g.add_edge(w, target)

    data_rw = from_networkx(g).to(device)

    # -------- copiar atributos originales --------
    data_rw.x = data.x.clone()
    if hasattr(data, "y"):
        data_rw.y = data.y.clone()

    return data_rw
