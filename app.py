import os
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template, request, redirect, url_for, jsonify
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def build_graph_from_df(df):
    G = nx.DiGraph()
    df.columns = df.columns.str.lower().str.strip()

    # Case 1: user_id / post_id (with parent/retweet columns)
    if {'user_id', 'post_id'}.issubset(df.columns):
        for _, row in df.iterrows():
            user_node = str(row['user_id'])
            post_node = str(row['post_id'])
            
            # Label node attributes
            is_misinfo = int(row['label']) if 'label' in df.columns else 0
            
            G.add_node(user_node, label='User', misinfo=0)
            G.add_node(post_node, label='Post', misinfo=is_misinfo)
            G.add_edge(user_node, post_node)
            
            for parent_col in ['retweeted_from', 'parent_post', 'source_post', 'retweet_id']:
                if parent_col in df.columns and pd.notna(row[parent_col]):
                    parent_node = str(row[parent_col])
                    G.add_node(parent_node, label='Post', misinfo=is_misinfo)
                    G.add_edge(parent_node, post_node)
                    break
    
    # Case 2: source / target (common for network edge lists)
    elif {'source', 'target'}.issubset(df.columns):
        for _, row in df.iterrows():
            src = str(row['source'])
            tgt = str(row['target'])
            is_misinfo = int(row['label']) if 'label' in df.columns else 0
            
            G.add_node(src, misinfo=is_misinfo)
            G.add_node(tgt, misinfo=is_misinfo)
            G.add_edge(src, tgt)
    
    # Case 3: tweet_ids (space separated chains)
    elif 'tweet_ids' in df.columns:
        for _, row in df.iterrows():
            nodes = str(row['tweet_ids']).split()
            is_misinfo = int(row['label']) if 'label' in df.columns else 0
            for i in range(len(nodes) - 1):
                G.add_node(nodes[i], misinfo=is_misinfo)
                G.add_node(nodes[i+1], misinfo=is_misinfo)
                G.add_edge(nodes[i], nodes[i+1])
    
    # Case 4: Fallback for any 2-column CSV
    elif len(df.columns) >= 2:
        col1, col2 = df.columns[0], df.columns[1]
        for _, row in df.iterrows():
            u, v = str(row[col1]), str(row[col2])
            is_misinfo = int(row['label']) if 'label' in df.columns else 0
            G.add_node(u, misinfo=is_misinfo)
            G.add_node(v, misinfo=is_misinfo)
            G.add_edge(u, v)
            
    return G

def run_misinfo_analysis(G):
    # Node features
    degrees = dict(G.degree())
    bet_cen = nx.betweenness_centrality(G)

    # Use the labels we extracted from the dataset
    predictions = []
    for node in G.nodes():
        # If the dataset had a label, use it. 
        # Otherwise, fallback to a structure-based guess.
        is_misinfo = G.nodes[node].get('misinfo', 0)
        
        # If no label, we can use a heuristic (but prioritize label)
        if 'misinfo' not in G.nodes[node]:
            deg_norm = degrees[node] / len(G.nodes()) if len(G.nodes()) > 0 else 0
            cen_norm = bet_cen[node]
            if deg_norm > 0.01 and cen_norm > 0.01:
                is_misinfo = 1
        
        predictions.append(is_misinfo)

    return predictions, list(G.nodes()), bet_cen

def create_interactive_graph(G, predictions, node_list):
    pos = nx.spring_layout(G, seed=42) # Seed for consistency
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    # Map predictions to colors
    color_map = {0: '#2ecc71', 1: '#e74c3c'} # Green for real, Red for misinfo
    
    for i, node in enumerate(node_list):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        pred = predictions[i] if predictions is not None else 0
        node_color.append(color_map[pred])
        
        # Hover info
        deg = G.degree(node)
        status = "Misinformation" if pred == 1 else "Real"
        node_text.append(f'Node: {node}<br>Degree: {deg}<br>Status: {status}')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=15,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict(text='Network Propagation Graph', font=dict(size=16)),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    
    return fig.to_json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Read dataset with robust encoding detection
        try:
            # Try common encodings
            for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # Fallback to default if all fail
                df = pd.read_csv(filepath)
        except Exception as e:
            return f"Error reading CSV file: {str(e)}", 400
        
        # Build graph
        G = build_graph_from_df(df)
        
        if len(G.nodes()) == 0:
            return "Error: Could not construct a valid graph from the dataset.", 400
            
        # Analysis
        predictions, node_list, betweenness = run_misinfo_analysis(G)
        
        # Calculate Degree Centrality for visualization and table
        deg_centrality = nx.degree_centrality(G)
        
        # Metrics
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
        density = nx.density(G)
        clustering = nx.average_clustering(G.to_undirected()) if num_nodes > 2 else 0
        misinfo_count = int(sum(predictions)) if predictions is not None else 0
        
        # Threshold: if more than 50% are misinformation, it's a high spread
        is_misinfo_spread = misinfo_count > (num_nodes * 0.5) 
        
        # Suspicious spreaders (nodes labeled as misinformation)
        degrees = dict(G.degree())
        suspicious_nodes = []
        if predictions is not None:
            for i, node in enumerate(node_list):
                if predictions[i] == 1:
                    suspicious_nodes.append({
                        'id': node,
                        'degree': degrees[node],
                        'centrality': round(deg_centrality[node], 4)
                    })
        
        # Sort suspicious spreaders by degree
        suspicious_nodes = sorted(suspicious_nodes, key=lambda x: x['degree'], reverse=True)[:10]
        
        # Centrality Visualization (Top 10)
        top_centrality = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        cent_fig = px.bar(
            x=[str(x[0]) for x in top_centrality], 
            y=[x[1] for x in top_centrality],
            labels={'x': 'Node ID', 'y': 'Degree Centrality'},
            title="Top 10 Nodes by Degree Centrality"
        )
        
        results = {
            'dataset_name': file.filename,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': round(avg_degree, 2),
            'density': round(density, 4),
            'clustering': round(clustering, 4),
            'misinfo_count': misinfo_count,
            'is_misinfo_spread': is_misinfo_spread,
            'suspicious_nodes': suspicious_nodes,
            'graph_json': create_interactive_graph(G, predictions, node_list),
            'centrality_json': cent_fig.to_json()
        }
        
        return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
