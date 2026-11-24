"""
ADVANCED MULTIMODAL SCENE ANALYTICS DASHBOARD
Vision-Language Joint Framework for Multimodal Scene Understanding

Author: AYANSH VERMA (2403A54025)
Course: 24BTCAI CSB18
"""

import os
import sys
import json
import io
import networkx as nx
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_FRONTEND_DIR)
for _p in [_ROOT_DIR, _FRONTEND_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.inference import MultimodalInferenceEngine
from src.data_loader import build_synthetic_catalog

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Scene Analytics Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp { background: #060C18; color: #F1F5F9; }

/* Hero */
.hero {
  background: linear-gradient(135deg,rgba(15,23,42,.95) 0%,rgba(30,41,59,.7) 100%);
  border: 1px solid rgba(56,189,248,.2);
  border-radius: 16px;
  padding: 22px 28px;
  margin-bottom: 22px;
  box-shadow: 0 10px 40px -10px rgba(0,0,0,.7);
}
.hero-title {
  font-size: 24px; font-weight: 800;
  background: linear-gradient(90deg, #F8FAFC 0%, #38BDF8 50%, #818CF8 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 4px; letter-spacing: -.5px;
}
.hero-sub { color: #94A3B8; font-size: 13px; margin-bottom: 14px; }
.pills { display: flex; flex-wrap: wrap; gap: 7px; }
.pill {
  padding: 4px 11px; font-size: 11px; font-weight: 700;
  border-radius: 20px; border: 1px solid rgba(255,255,255,.1);
}
.p-blue   { background:rgba(56,189,248,.12); color:#38BDF8; border-color:rgba(56,189,248,.35); }
.p-indigo { background:rgba(99,102,241,.12);  color:#818CF8; border-color:rgba(99,102,241,.3); }
.p-green  { background:rgba(16,185,129,.12);  color:#34D399; border-color:rgba(16,185,129,.3); }
.p-slate  { background:rgba(148,163,184,.1);  color:#CBD5E1; }

/* Cards */
.card {
  background: rgba(15,23,42,.75);
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 13px;
  padding: 18px;
  margin-bottom: 16px;
  box-shadow: 0 4px 20px rgba(0,0,0,.3);
}
.card-title {
  font-size: 14px; font-weight: 700; color: #E2E8F0;
  margin-bottom: 12px; display: flex; align-items: center; gap: 7px;
}

/* Caption boxes */
.cap-sota {
  background: linear-gradient(135deg,rgba(6,182,212,.1) 0%,rgba(15,23,42,.8) 100%);
  border-left: 4px solid #06B6D4;
  border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
}
.cap-base {
  background: rgba(30,41,59,.4);
  border-left: 4px solid #475569;
  border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;
}
.cap-text { font-size: 17px; font-weight: 700; color: #F8FAFC; line-height: 1.4; margin: 4px 0 7px; }
.cap-lbl  { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .7px; }

/* Metric tiles */
.mets { display:flex; gap:10px; margin-top:12px; }
.met  {
  flex:1; background:rgba(30,41,59,.5);
  border:1px solid rgba(255,255,255,.06);
  border-radius:10px; padding:12px; text-align:center;
}
.met-v { font-size:18px; font-weight:800; color:#38BDF8; font-family:'JetBrains Mono',monospace; }
.met-l { font-size:11px; font-weight:600; color:#94A3B8; margin-top:2px; }

/* Attribute badge */
.abadge {
  display:inline-block;
  background:rgba(56,189,248,.14); color:#38BDF8;
  border:1px solid rgba(56,189,248,.3);
  padding:2px 8px; border-radius:12px;
  font-size:11px; font-weight:600;
  margin-right:4px; margin-bottom:4px;
}

/* Detection card */
.det-card {
  background:rgba(15,23,42,.7);
  border:1px solid rgba(255,255,255,.07);
  border-radius:10px; padding:12px; margin-bottom:9px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap:7px; background:transparent;
  border-bottom:1px solid rgba(255,255,255,.08);
  padding-bottom:5px;
}
.stTabs [data-baseweb="tab"] {
  border-radius:7px; padding:7px 14px;
  font-weight:700; font-size:12.5px; color:#94A3B8;
  background:transparent;
}
.stTabs [aria-selected="true"] {
  color:#38BDF8 !important;
  background:rgba(56,189,248,.1) !important;
  border:1px solid rgba(56,189,248,.3) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #0B1120; }
section[data-testid="stSidebar"] .stMarkdown { color: #CBD5E1; }
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising Vision-Language Framework…")
def _load_engine():
    return MultimodalInferenceEngine()

@st.cache_data
def _load_catalog():
    img_dir = os.path.join(_ROOT_DIR, "data", "catalog_images")
    dog_path = r"C:\Users\ayans\Downloads\download.png"
    return build_synthetic_catalog(img_dir, user_dog_path=dog_path)

engine  = _load_engine()
catalog = _load_catalog()

# ── Hero ──────────────────────────────────────────────────────────────────────
import torch
_device = "CUDA" if torch.cuda.is_available() else "CPU"
st.markdown(f"""
<div class="hero">
  <div class="hero-title">🧠 Advanced Multimodal Scene Analytics Dashboard</div>
  <div class="hero-sub">Vision-Language Joint Framework for Multimodal Scene Understanding</div>
  <div class="pills">
    <span class="pill p-blue">👤 Ayansh Verma · 2403A54025</span>
    <span class="pill p-indigo">📚 24BTCAI CSB18</span>
    <span class="pill p-green">🔬 BLIP·YOLOS·CLIP·BLIP-VQA</span>
    <span class="pill p-slate">⚡ {_device}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Controls")
    st.markdown("---")

    # Image source
    src = st.radio("Image Source:", ["Benchmark Scenes", "Upload Image"], index=0)
    active_image = None
    active_meta  = None

    if src == "Benchmark Scenes":
        titles = [v.get("title", f"Scene {k}") for k, v in catalog.items()]
        sel_title = st.selectbox("Select Scene:", titles)
        active_meta = next(v for v in catalog.values() if v.get("title") == sel_title)
        active_image = Image.open(active_meta["image_path"]).convert("RGB")
    else:
        up = st.file_uploader("Upload Image (JPG/PNG):", type=["jpg","jpeg","png"])
        if up:
            active_image = Image.open(up).convert("RGB")
            active_meta  = None
        else:
            first = next(iter(catalog.values()))
            active_image = Image.open(first["image_path"]).convert("RGB")
            active_meta  = first
            st.info("No file uploaded. Showing default scene.")

    st.markdown("---")
    st.markdown("#### ⚙️ Parameters")
    det_thr = st.slider("Detection Threshold", 0.10, 0.80, 0.25, 0.05)
    cap_bms = st.slider("Caption Beam Width", 1, 4, 1, help="Set to 1 for fastest inference (<1s); 2-4 for beam search.")
    qa_q    = st.text_input("💬 Ask a Question:", placeholder="e.g. What color is the dog?")

    st.markdown("---")
    st.caption("⚡ Accelerated with batch ROI embeddings & fast token decoding.")

# ── Inference ─────────────────────────────────────────────────────────────────
with st.spinner("⚡ Running Multimodal Vision-Language Pipeline…"):
    catalog_list = [{"image_path": v["image_path"],
                     "title": v.get("title", "Scene"),
                     "image_caption": v.get("image_caption", "")}
                    for v in catalog.values()]
    results = engine.process(
        active_image,
        question=qa_q,
        retrieval_catalog=catalog_list,
        det_threshold=det_thr,
        caption_beams=cap_bms,
    )

detections = results["enhanced_detections"]
triplets   = results["triplets"]
blip_res   = results["blip_caption"]
gpt2_res   = results["baseline_caption"]
det_res    = results["detection"]
vqa_res    = results["vqa"]
sg_img     = results["scene_graph_image"] if "scene_graph_image" in results else results.get("scene_graph_figure")
sg_json    = results["scene_graph_json"]
retrieval  = results["retrieval"]

# ── Main tabs ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📷 1. Image & Captions",
    "🎯 2. Objects & Attributes",
    "🕸️ 3. Scene Graph",
    "❓ 4. Visual Q&A",
    "🔍 5. Image Retrieval",
    "📊 6. Evaluation & Diagnostics",
])

# ───── Tab 1: Image & Captions ────────────────────────────────────────────────
with tabs[0]:
    c_img, c_cap = st.columns([1.1, 1])
    with c_img:
        st.markdown('<div class="card-title">📷 Input Visual Scene</div>', unsafe_allow_html=True)
        view = st.radio("View:", ["Original", "Object Detections Overlay"], horizontal=True)
        st.image(results["annotated_image"] if view == "Object Detections Overlay" else active_image,
                 use_container_width=True)
        if active_meta and active_meta.get("region_descriptions"):
            st.markdown("**Visual Genome Region Descriptions:**")
            for p in active_meta["region_descriptions"][:4]:
                st.caption(f"• {p}")

    with c_cap:
        st.markdown('<div class="card-title">📝 Generated Descriptions</div>', unsafe_allow_html=True)
        # BLIP
        st.markdown(f"""
        <div class="cap-sota">
          <div class="cap-lbl" style="color:#06B6D4;">✨ SOTA — BLIP (ViT-B + Transformer · Visual Genome)</div>
          <div class="cap-text">"{blip_res['caption']}"</div>
          <div style="font-size:12px;color:#94A3B8;">Trained on Visual Genome • Latency: <b style="color:#06B6D4;">{blip_res['latency_s']}s</b></div>
        </div>
        """, unsafe_allow_html=True)
        # Baseline
        st.markdown(f"""
        <div class="cap-base">
          <div class="cap-lbl" style="color:#64748B;">🏛️ Baseline — ViT-GPT2 (Known for COCO bias & hallucinations)</div>
          <div class="cap-text" style="font-size:15px;color:#CBD5E1;">"{gpt2_res['caption']}"</div>
          <div style="font-size:11px;color:#475569;">COCO Baseline • Latency: {gpt2_res['latency_s']}s</div>
        </div>
        """, unsafe_allow_html=True)

        # Full Multimodal Scene Synthesis
        if detections and triplets:
            obj_summary = ", ".join([f"{d['label']} ({d['attributes'][0]['attribute'] if d['attributes'] else ''})" for d in detections])
            rel_summary = "; ".join([f"{t['subject']} is {t['predicate']} {t['object']}" for t in triplets])
            full_description = f"The scene features {obj_summary}. Spatial interactions indicate that {rel_summary}."
            st.markdown(f"""
            <div class="card" style="margin-top: 10px; border-left: 4px solid #10B981; background: rgba(16, 185, 129, 0.08);">
              <div class="cap-lbl" style="color:#10B981;">🌐 Comprehensive Scene Understanding Synthesis</div>
              <div style="font-size:14px; font-weight:600; color:#E2E8F0; margin-top:5px; line-height:1.5;">{full_description}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="mets">
          <div class="met"><div class="met-v">{blip_res['latency_s']}s</div><div class="met-l">BLIP Latency</div></div>
          <div class="met"><div class="met-v">{cap_bms}</div><div class="met-l">Beam Width</div></div>
          <div class="met"><div class="met-v">{results['total_latency_s']}s</div><div class="met-l">Total Pipeline</div></div>
        </div>
        """, unsafe_allow_html=True)

# ───── Tab 2: Objects & Attributes ───────────────────────────────────────────
with tabs[1]:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown('<div class="card-title">🎯 YOLOS-ViT Detections</div>', unsafe_allow_html=True)
        st.image(results["annotated_image"], use_container_width=True)
        st.caption(f"Detection threshold: {det_thr} · Latency: {det_res['latency_s']}s")
    with c2:
        st.markdown(f'<div class="card-title">🏷️ Entities & CLIP Attributes ({len(detections)} objects)</div>',
                    unsafe_allow_html=True)
        if detections:
            for det in detections:
                attr_html = "".join([
                    f'<span class="abadge">{a["attribute"]} ({int(a["confidence"]*100)}%)</span>'
                    for a in det["attributes"]
                ])
                st.markdown(f"""
                <div class="det-card">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="font-size:14px;font-weight:800;color:#F8FAFC;">#{det['id']} {det['label'].upper()}</span>
                    <span class="pill p-blue">Conf: {int(det['confidence']*100)}%</span>
                  </div>
                  <div style="font-size:11px;color:#94A3B8;margin-bottom:6px;">Box: <code>{det['box']}</code></div>
                  <div>{attr_html}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No objects above threshold. Try lowering the Detection Threshold slider.")

        if detections:
            df = pd.DataFrame([{
                "ID": f"#{d['id']}",
                "Entity": d["label"].upper(),
                "Conf %": f"{int(d['confidence']*100)}",
                "Top Attribute": d["attributes"][0]["attribute"] if d["attributes"] else "-",
                "Box [x1,y1,x2,y2]": str(d["box"])
            } for d in detections])
            st.dataframe(df, use_container_width=True, hide_index=True)

# ───── Tab 3: Scene Graph ─────────────────────────────────────────────────────
with tabs[2]:
    c_sg, c_tr = st.columns([1.2, 1])
    with c_sg:
        st.markdown('<div class="card-title">🕸️ Semantic Scene Graph (NetworkX)</div>', unsafe_allow_html=True)
        if sg_img is not None:
            st.image(sg_img, use_container_width=True)
        else:
            st.info("No scene graph generated.")

    with c_tr:
        st.markdown('<div class="card-title">🔗 Relational Triplets</div>', unsafe_allow_html=True)
        G = results["scene_graph"]
        if triplets:
            df_t = pd.DataFrame([
                {"Subject": t["subject"].upper(),
                 "Predicate": f"→ {t['predicate'].upper()} →",
                 "Object": t["object"].upper(),
                 "Conf %": f"{int(t['confidence']*100)}"}
                for t in triplets
            ])
            st.dataframe(df_t, use_container_width=True, hide_index=True)

            if len(detections) == 1:
                st.info("ℹ️ Only one entity detected. The scene graph connects it to its environment context.")
        else:
            st.info("No relational triplets generated for this scene.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes", len(G.nodes))
        col2.metric("Edges", len(G.edges))
        col3.metric("Density", f"{nx.density(G):.2f}" if len(G.nodes) > 1 else "1.00")

        st.download_button(
            "⬇️ Download Scene Graph JSON",
            data=sg_json,
            file_name="scene_graph.json",
            mime="application/json",
        )

# ───── Tab 4: Visual Q&A ─────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="card-title">❓ Visual Question Answering (BLIP-VQA)</div>', unsafe_allow_html=True)
    c_qa1, c_qa2 = st.columns([1, 1.2])
    with c_qa1:
        st.image(active_image, use_container_width=True)
    with c_qa2:
        from src.vqa import BLIPVQAModel
        st.markdown("**Ask a question about this image:**")
        q_input = st.text_input("Question:", value=qa_q or "What animal is in the image?", key="qa_text")
        if st.button("Ask AI 🤖", type="primary"):
            engine._load_vqa()
            vqa_live = engine._vqa.answer(active_image, q_input)
            st.markdown(f"""
            <div class="cap-sota" style="margin-top:14px;">
              <div class="cap-lbl" style="color:#06B6D4;">🤖 BLIP-VQA Answer</div>
              <div class="cap-text">Q: {vqa_live['question']}</div>
              <div class="cap-text" style="color:#38BDF8;">A: {vqa_live['answer']}</div>
              <div style="font-size:11px;color:#94A3B8;">Latency: {vqa_live['latency_s']}s</div>
            </div>
            """, unsafe_allow_html=True)
        elif vqa_res and vqa_res.get("question"):
            st.markdown(f"""
            <div class="cap-sota" style="margin-top:14px;">
              <div class="cap-lbl" style="color:#06B6D4;">🤖 BLIP-VQA Answer</div>
              <div class="cap-text">Q: {vqa_res['question']}</div>
              <div class="cap-text" style="color:#38BDF8;">A: {vqa_res['answer']}</div>
              <div style="font-size:11px;color:#94A3B8;">Latency: {vqa_res['latency_s']}s</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Enter a question in the sidebar or the box above, then click **Ask AI**.")

        st.markdown("**Sample Questions:**")
        for q in BLIPVQAModel.SAMPLE_QUESTIONS[:5]:
            st.caption(f"• {q}")

# ───── Tab 5: Image Retrieval ─────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="card-title">🔍 Zero-Shot Cross-Modal Retrieval (CLIP ViT)</div>', unsafe_allow_html=True)
    sample_queries = [
        "a white fluffy dog sitting on grass",
        "a red car on the road",
        "a blue sofa in a living room",
        "a dining table with food",
    ]
    q_sel = st.selectbox("Sample Queries:", sample_queries)
    q_custom = st.text_input("Or type custom query:")
    final_q = q_custom.strip() if q_custom.strip() else q_sel

    if st.button("🔍 Search", type="primary"):
        engine._load_retrieval(catalog_list)
        search_results = engine._retrieval.search(final_q, catalog=catalog_list, top_k=min(4, len(catalog_list)))
        st.markdown(f"**Results for:** *'{final_q}'*")
        cols = st.columns(len(search_results))
        for i, r in enumerate(search_results):
            with cols[i]:
                st.image(r["image_path"], use_container_width=True)
                st.caption(f"**#{r['rank']}** {r['title']}")
                st.metric("Similarity", f"{r['similarity']*100:.1f}%")
    elif retrieval:
        st.markdown(f"**Auto-retrieval for caption:** *'{blip_res['caption']}'*")
        cols = st.columns(len(retrieval))
        for i, r in enumerate(retrieval):
            with cols[i]:
                st.image(r["image_path"], use_container_width=True)
                st.caption(f"#{r['rank']} {r['title']}")
                st.metric("Similarity", f"{r['similarity']*100:.1f}%")

# ───── Tab 6: Evaluation & Diagnostics ───────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="card-title">📊 Architecture Verification & Model Diagnostics</div>',
                unsafe_allow_html=True)
    import platform, torch, transformers
    st.code(f"""
System       : {platform.platform()}
Python       : {platform.python_version()}
PyTorch      : {torch.__version__}
Transformers : {transformers.__version__}
CUDA         : {torch.cuda.is_available()}
Device       : {_device}
Total Latency: {results['total_latency_s']}s
""")

    st.markdown("#### Architecture Alignment vs Project Specification")
    df_arch = pd.DataFrame([
        {"Component": "Captioner", "Spec Requirement": "ViT Encoder + Transformer Decoder", "Implementation": "BLIP (ViT-B/16 + Transformer, VG fine-tuned)", "Status": "✅"},
        {"Component": "Baseline",  "Spec Requirement": "ViT Captioner Comparison",          "Implementation": "ViT-GPT2 (nlpconnect)",                         "Status": "✅"},
        {"Component": "Detector",  "Spec Requirement": "ViT-based Object Detector",          "Implementation": "YOLOS-tiny (pure ViT, no CNN)",                  "Status": "✅"},
        {"Component": "Attributes","Spec Requirement": "Multi-label per-object classifier",  "Implementation": "CLIP ViT Zero-Shot ROI Projection",              "Status": "✅"},
        {"Component": "Relations", "Spec Requirement": "Subject-Predicate-Object Triplets",  "Implementation": "CLIP pairwise + env-grounding",                  "Status": "✅"},
        {"Component": "Scene Graph","Spec Requirement":"NetworkX directed graph",             "Implementation": "NetworkX DiGraph + Matplotlib renderer",          "Status": "✅"},
        {"Component": "VQA",       "Spec Requirement": "Visual Question Answering",           "Implementation": "BLIP-VQA-base (Salesforce)",                     "Status": "✅"},
        {"Component": "Retrieval", "Spec Requirement": "CLIP image-text retrieval",           "Implementation": "CLIP cosine similarity ranking",                 "Status": "✅"},
    ])
    st.table(df_arch)

    if active_meta and active_meta.get("region_descriptions"):
        st.markdown("#### Ground-Truth Annotations for Selected Scene")
        st.write("**Region Descriptions:**", active_meta["region_descriptions"])
        if active_meta.get("relationships"):
            st.write("**Ground-Truth Relationships:**")
            for rel in active_meta["relationships"][:5]:
                pred = rel.get("predicate", "?")
                subj = rel.get("subject", {}).get("name", rel.get("subject_id", "?"))
                obj  = rel.get("object", {}).get("name", rel.get("object_id", "?"))
                st.caption(f"• ({subj}) → [{pred}] → ({obj})")


# Fix missing import used in scene graph tab
def import_nx_density(G):
    import networkx as nx
    return round(nx.density(G), 2) if len(G.nodes) > 1 else 1.0
