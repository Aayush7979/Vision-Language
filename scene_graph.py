"""
Scene Graph Generation Module
Builds a NetworkX directed graph from detected entities and predicted relationships.
Renders a publication-quality dark-theme visualization.
"""

import io
import logging
import json
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from PIL import Image

logger = logging.getLogger(__name__)

# Color scheme
ENTITY_COLORS = {
    "living": "#38BDF8",   # Cyan  — animals / people
    "env":    "#10B981",   # Emerald — environment contexts
    "object": "#818CF8",   # Indigo — inanimate objects
}
LIVING_LABELS = {"dog", "cat", "person", "man", "woman", "bird", "horse", "cow", "sheep",
                 "elephant", "bear", "zebra", "giraffe", "sheep", "mouse", "rabbit"}
ENV_LABELS    = {"grass", "lawn", "road", "floor", "table", "sky", "water", "pavement",
                 "carpet", "shelf", "ground", "wall", "ceiling"}


def _node_color(label: str) -> str:
    lbl = label.lower()
    if lbl in LIVING_LABELS:
        return ENTITY_COLORS["living"]
    if lbl in ENV_LABELS:
        return ENTITY_COLORS["env"]
    return ENTITY_COLORS["object"]


def build_scene_graph(
    enhanced_detections: List[Dict],
    triplets: List[Dict],
) -> nx.DiGraph:
    """
    Construct a directed scene graph.
    Nodes: detected entities (and any context-environment nodes from single-entity triplets).
    Edges: predicted relational triplets.
    """
    G = nx.DiGraph()

    # Add entity nodes
    for det in enhanced_detections:
        node_key = f"{det['label']}_{det.get('id', 0)}"
        attr_str = ", ".join([a["attribute"] for a in det.get("attributes", [])[:2]])
        G.add_node(
            node_key,
            label=det["label"],
            attributes=attr_str,
            confidence=det.get("confidence", 0),
            box=det.get("box", []),
            object_id=det.get("id", 0),
        )

    # Add context-environment nodes that may come from single-entity grounding
    for t in triplets:
        subj_key = f"{t['subject']}_{t['subject_id']}"
        obj_key  = f"{t['object']}_{t['object_id']}"
        if not G.has_node(subj_key):
            G.add_node(subj_key, label=t["subject"], attributes="", confidence=1.0, box=[], object_id=t["subject_id"])
        if not G.has_node(obj_key):
            G.add_node(obj_key,  label=t["object"],  attributes="", confidence=1.0, box=[], object_id=t["object_id"])
        G.add_edge(
            subj_key, obj_key,
            predicate=t["predicate"],
            confidence=t["confidence"],
        )

    return G


def render_scene_graph(
    G: nx.DiGraph,
    title: str = "Visual Genome Semantic Scene Graph",
    figsize: Tuple = (9, 6),
    dpi: int = 150,
) -> plt.Figure:
    """
    Render the directed scene graph with dark background,
    color-coded nodes (living/env/object), edge predicates.
    Returns a matplotlib Figure (not shown; caller saves or displays).
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("#0B1120")
    fig.patch.set_facecolor("#0B1120")

    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, "No entities detected.", color="#64748B",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.axis("off")
        return fig

    pos = nx.spring_layout(G, k=1.8, seed=42)
    colors = [_node_color(G.nodes[n].get("label", "")) for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3000, node_color=colors,
                           edgecolors="#FFFFFF", linewidths=1.6)

    # Node labels: LABEL\n[attrs]
    node_labels = {}
    for n, data in G.nodes(data=True):
        lbl  = data.get("label", n).upper()
        attr = data.get("attributes", "")
        node_labels[n] = f"{lbl}\n[{attr.upper()}]" if attr else lbl

    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                            font_size=8, font_weight="bold", font_color="#0F172A")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#38BDF8", arrowstyle="-|>",
                           arrowsize=18, width=2.0, connectionstyle="arc3,rad=0.15")
    edge_labels = nx.get_edge_attributes(G, "predicate")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=8, font_color="#F8FAFC",
                                 bbox=dict(boxstyle="round,pad=0.25",
                                           fc="#1E293B", ec="#38BDF8", lw=1.1))
    ax.set_title(title, fontsize=11, fontweight="bold", color="#F8FAFC", pad=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


def figure_to_pil(fig: plt.Figure) -> Image.Image:
    """Convert a matplotlib Figure to a PIL Image without saving to disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).copy()


def scene_graph_to_json(G: nx.DiGraph) -> str:
    """Export the scene graph to a JSON string (nodes + edges)."""
    data = {
        "nodes": [
            {"id": n, **{k: v for k, v in d.items() if k not in ("box",)}}
            for n, d in G.nodes(data=True)
        ],
        "edges": [
            {"from": u, "to": v, **d}
            for u, v, d in G.edges(data=True)
        ],
    }
    return json.dumps(data, indent=2)

