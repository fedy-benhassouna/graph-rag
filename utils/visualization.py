import networkx as nx
import plotly.graph_objects as go
from config import (
    DEFAULT_NODE_SIZE,
    DEFAULT_NODE_COLOR,
    DEFAULT_EDGE_WIDTH,
    DEFAULT_EDGE_COLOR,
    DEFAULT_SPRING_K
)

def create_3d_graph(driver, node_size=DEFAULT_NODE_SIZE, node_color=DEFAULT_NODE_COLOR,
                   edge_width=DEFAULT_EDGE_WIDTH, edge_color=DEFAULT_EDGE_COLOR,
                   spring_k=DEFAULT_SPRING_K):
    """Generate 3D graph visualization using Plotly with customizable settings."""
    G = nx.DiGraph()
    
    with driver.session() as session:
        result = session.run("""
        MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
        RETURN e1.name as source, r.type as relationship, e2.name as target
        """)
        records = list(result)
        
    if not records:  # If no relationships exist
        return None
        
    for record in records:
        G.add_edge(record["source"], record["target"])
        
    pos = nx.spring_layout(G, dim=3, k=spring_k)
    
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edges_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=edge_width, color=edge_color),
        hoverinfo='none',
        mode='lines')

    node_x, node_y, node_z = [], [], []
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    nodes_trace = go.Scatter3d(
        x=node_x, y=node_y, z=edge_z,
        mode='markers',
        hovertext=list(G.nodes()),
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=node_color,
            line_width=2))

    fig = go.Figure(data=[edges_trace, nodes_trace])
    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig 