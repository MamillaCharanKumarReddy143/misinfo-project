from flask import Flask, render_template, request
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    graph_path = None
    error = None
    analysis = {}
    timestamp = None
    filename = None   # ✅ Added

    if request.method == 'POST':
        file = request.files.get('file')
        filename = file.filename   # ✅ ADD THIS

        if not file:
            error = "Please upload a CSV file"
            return render_template('index.html', error=error)

        filename = file.filename   # ✅ Added (store selected file name)

        df = pd.read_csv(file)
        df.columns = df.columns.str.lower()

        # -----------------------------
        # BASIC DATA ANALYSIS
        # -----------------------------

        total_posts = len(df)
        fake_posts = 0
        real_posts = 0
        fake_percentage = 0

        if 'label' in df.columns:
            fake_posts = len(df[df['label'].astype(str).str.lower() == 'fake'])
            real_posts = len(df[df['label'].astype(str).str.lower() == 'real'])

            if total_posts > 0:
                fake_percentage = round((fake_posts / total_posts) * 100, 2)

        analysis['total_posts'] = total_posts
        analysis['fake_posts'] = fake_posts
        analysis['real_posts'] = real_posts
        analysis['fake_percentage'] = fake_percentage

        # -----------------------------
        # PROPAGATION GRAPH
        # -----------------------------

        G = nx.DiGraph()

        if {'user_id','post_id'}.issubset(df.columns):
            for _, row in df.iterrows():
                user = row['user_id']
                post = row['post_id']
                parent = row.get('retweeted_from', None)

                G.add_node(user, type='user')
                G.add_node(post, type='post')
                G.add_edge(user, post)

                if pd.notna(parent):
                    G.add_edge(parent, post)

        elif 'tweet_ids' in df.columns:
            for _, row in df.iterrows():
                nodes = str(row['tweet_ids']).split()
                for i in range(len(nodes)):
                    G.add_node(nodes[i], type='tweet')
                    if i > 0:
                        G.add_edge(nodes[i-1], nodes[i])

        elif 'tweet_id' in df.columns and 'retweet_id' in df.columns:
            for _, r in df.iterrows():
                t1 = r['tweet_id']
                t2 = r['retweet_id']
                G.add_node(t1, type='tweet')
                G.add_node(t2, type='tweet')
                G.add_edge(t1, t2)

        else:
            error = "No tweet/retweet propagation columns found"
            return render_template('index.html', error=error)

        # -----------------------------
        # CENTRALITY METRICS
        # -----------------------------

        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)

        top_degree = sorted(degree_centrality.items(),
                            key=lambda x: x[1],
                            reverse=True)[:5]

        top_betweenness = sorted(betweenness_centrality.items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:5]

        analysis['top_degree'] = {k: round(v,4) for k,v in top_degree}
        analysis['top_betweenness'] = {k: round(v,4) for k,v in top_betweenness}

        # Suspicious detection
        suspicious_nodes = []
        if len(G.nodes()) > 0:
            degree_threshold = sum(degree_centrality.values()) / len(degree_centrality)
            betweenness_threshold = sum(betweenness_centrality.values()) / len(betweenness_centrality)

            for node in G.nodes():
                if (degree_centrality.get(node,0) > degree_threshold and
                    betweenness_centrality.get(node,0) > betweenness_threshold):
                    suspicious_nodes.append(node)

        analysis['suspicious_spreaders'] = suspicious_nodes
        analysis['suspicious_count'] = len(suspicious_nodes)

        # -----------------------------
        # DRAW GRAPH
        # -----------------------------

        pos = nx.spring_layout(G)
        node_colors = ['skyblue' if G.nodes[n].get('type')=='user' else 'red' for n in G.nodes()]

        plt.figure(figsize=(12,8))
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700)

        os.makedirs('static', exist_ok=True)
        graph_path = 'static/propagation_graph.png'
        plt.savefig(graph_path)
        plt.close()

        timestamp = time.time()

        result = {
            'nodes': len(G.nodes()),
            'edges': len(G.edges())
        }

    return render_template('index.html',
                           result=result,
                           graph=graph_path,
                           error=error,
                           analysis=analysis,
                           timestamp=timestamp,
                           filename=filename)   # ✅ Added

if __name__ == "__main__":
    app.run(debug=True)
