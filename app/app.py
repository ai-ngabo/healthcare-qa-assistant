"""AfriHealth Knowledge QA Assistant -  Web App.

How it works
1. The question is turned into a vector by the fine-tuned Transformer (Notebook 03).
2. It is compared (cosine similarity) with the stored vectors of all knowledge-base answers.
3. Safety checks decide whether to answer, refuse, or ask for a condition name.
4. The best matching NIH answer is shown together with the question it answers.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Healthcare QA Assistant", page_icon="🩺", layout="centered",
                   initial_sidebar_state="collapsed")

TOP_K = 3
PREVIEW_CHARS = 700

# Example questions for the user to try
EXAMPLES = [
    "What are the symptoms of Diabetes?",
    "How is Glaucoma treated?",
    "What causes Anemia?",
    "How can Asthma be prevented?",
]

# Hugging Face token (the model repo is public, so the app also works without it)
def setup_hf_token():
    # Loads the token from the hidden files.
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            token = st.secrets.get("HF_TOKEN")
        except Exception:
            token = None
    if token:
        os.environ["HF_TOKEN"] = str(token)


# Load settings, knowledge base and model (cached and loaded once per server)
def resolve_model_source(config):
    # model's path -> ./model folder -> Hugging Face Hub repo from config.json
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return env_path
    local = APP_DIR / "model"
    if (local / "modules.json").exists():
        return str(local)
    return config["model_hub_repo"]


@st.cache_resource(show_spinner="Loading the model. The first start can take a minute...")
def load_resources():
    with open(APP_DIR / "config.json") as f:
        config = json.load(f)

    kb = pd.read_csv(APP_DIR / "kb.csv")
    kb_emb = np.load(APP_DIR / "kb_embeddings.npy").astype("float32")
    kb_emb /= np.linalg.norm(kb_emb, axis=1, keepdims=True) + 1e-9
    if len(kb) != len(kb_emb):
        raise ValueError(f"kb.csv has {len(kb)} rows but kb_embeddings.npy has {len(kb_emb)}. Export them together.")

    from sentence_transformers import SentenceTransformer
    source = resolve_model_source(config)
    model = SentenceTransformer(source, device="cpu")
    model.max_seq_length = int(config["max_len"])
    return config, kb, kb_emb, model, source


setup_hf_token()
try:
    config, kb, kb_emb, model, MODEL_SOURCE = load_resources()
except Exception as error:
    st.error(f"The assistant could not start: {error}")
    st.stop()

THRESH = float(config["threshold"])  # out of domain when below the similarity score
MIN_NAME_OVERLAP = float(config.get("min_name_overlap", 0.5)) # share of the matched condition name found in the question


# Retrieval and safety checks
STOP = {"what", "are", "is", "the", "of", "for", "to", "how", "who", "at", "risk", "symptoms", "symptom",
        "treatment", "treatments", "causes", "cause", "diagnose", "tests", "exams", "outlook", "prevent",
        "prevention", "genetic", "changes", "related", "inherited", "many", "people", "affected", "does",
        "do", "a", "an", "in", "and", "or", "with", "i", "know", "about", "information", "need", "go",
        "find", "more", "stages", "complications", "considerations", "research", "being", "done", "new",
        "frequency", "susceptibility", "there", "can", "be", "by", "it", "this", "that", "s",
        "tell", "me", "my", "treated", "treat", "prevented", "caused", "signs", "please"}

GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you)\b[\s!.,?]*$", re.I)
SELF_HARM = re.compile(r"suicid|kill myself|end my life|want to die|self[- ]?harm|hurt myself", re.I)
URGENT = re.compile(r"chest pain|can'?t breathe|cannot breathe|trouble breathing|difficulty breathing|"
                    r"unconscious|severe bleeding|overdose|poison|seizure|heart attack|stroke", re.I)

WELCOME = ("Hello. I answer health questions using verified information from NIH sources. "
           "**Name the condition** in your question, for example *\"What are the symptoms of Glaucoma?\"* "
           "or *\"How is Diabetes treated?\"*.\n\n"
           "I give general information only. I cannot diagnose or recommend treatment.")
DISCLAIMER = "General information from NIH sources. Not a diagnosis or medical advice. See a health professional for personal concerns."
EMERGENCY_NOTE = ("If this may be an emergency, contact your local emergency service or go to the nearest "
                  "health facility now.")
SELF_HARM_REPLY = ("I'm sorry you are going through this. I can't help with this in a chat, but you do not have "
                   "to face it alone. Please reach out to someone you trust, or contact a local crisis line or "
                   "emergency service now. If you are in immediate danger, call your local emergency number.")


def topic_tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


def escape_md(text):
    # answers are plain text: stop characters such as * _ # | $ from being rendered as formatting
    return re.sub(r"([*_`#|<>$~\[\]])", r"\\\1", text)


def preview(text, limit=PREVIEW_CHARS):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("\n"))
    if end > limit * 0.6:
        cut = cut[:end + 1]
    return cut.rstrip() + " ..."


def search(query, top_k=TOP_K):
    q = model.encode([config["prefix"] + query], normalize_embeddings=True)[0].astype("float32")
    sims = kb_emb @ q
    idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in idx]


def name_overlap(query, matched_question):
    # share of the matched condition name that appears in the user's question
    name_tokens = topic_tokens(matched_question)
    if not name_tokens:
        return 1.0
    return len(topic_tokens(query) & name_tokens) / len(name_tokens)


def build_reply(query):
    # returns a dict that render_reply() draws; "kind" is message, topics or answer
    query = (query or "").strip()
    if not query:
        return {"kind": "message", "text": "Please type a question, for example *\"What are the symptoms of Glaucoma?\"*."}
    if GREETING.match(query):
        return {"kind": "message", "text": WELCOME}
    if SELF_HARM.search(query):
        return {"kind": "message", "text": SELF_HARM_REPLY}

    emergency = bool(URGENT.search(query))
    hits = search(query)
    best_i, best_sim = hits[0]

    # 1) Not a health question in my knowledge base
    if best_sim < THRESH:
        return {"kind": "message", "emergency": emergency,
                "text": "I can only answer health questions covered by my NIH knowledge base, and I could not "
                        "find a good match for this one. Try asking about a specific condition, for example "
                        "*\"What causes Anemia?\"*."}

    best_question = kb["question"].iloc[best_i]

    # 2) A match exists, but the user did not name the condition (for example a list of symptoms)
    if name_overlap(query, best_question) < MIN_NAME_OVERLAP:
        return {"kind": "topics", "emergency": emergency,
                "text": "I could not find a specific condition in your question. I am not a symptom checker and "
                        "cannot tell you what condition you may have.\n\n"
                        "If you know the condition, include its name. These are the closest topics I found:",
                "topics": [kb["question"].iloc[i] for i, _ in hits]}

    # 3) Answer
    return {"kind": "answer", "emergency": emergency,
            "matched_question": best_question, "score": best_sim,
            "strong": best_sim >= THRESH + 0.15,
            "answer": kb["answer"].iloc[best_i],
            "related": [(kb["question"].iloc[i], s) for i, s in hits[1:]]}


def render_reply(reply):
    if reply.get("emergency"):
        st.warning(EMERGENCY_NOTE, icon="🚨")
    kind = reply["kind"]
    if kind == "message":
        st.markdown(reply["text"])
    elif kind == "topics":
        st.markdown(reply["text"])
        for question in reply["topics"]:
            st.markdown(f"- {escape_md(question)}")
        st.caption(DISCLAIMER)
    else:
        st.caption("MATCHED QUESTION")
        st.markdown(f"**{escape_md(reply['matched_question'])}**")
        if reply["strong"]:
            st.badge(f"Strong match · {reply['score']:.2f}", icon=":material/check_circle:", color="green")
        else:
            st.badge(f"Possible match · {reply['score']:.2f}", icon=":material/help:", color="orange")
        answer = reply["answer"]
        st.markdown(escape_md(preview(answer)))
        if len(answer) > PREVIEW_CHARS:
            with st.expander("Read the full answer"):
                st.markdown(escape_md(answer))
        if reply["related"]:
            with st.expander("Related questions"):
                for question, score in reply["related"]:
                    st.markdown(f"- {escape_md(question)} ({score:.2f})")
        st.caption(DISCLAIMER)


def peak_memory_mb():
    # peak memory of this process (Linux only, shown in the technical info box)
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except Exception:
        return None


# User interface
st.markdown(
    f"""
    <style>
    .hero {{ background: linear-gradient(135deg, #0f766e 0%, #0e7490 100%); color: #ffffff; padding: 26px 28px;
             border-radius: 18px; margin-bottom: 18px; }}
    .hero h1 {{ margin: 0 0 6px 0; padding: 0; font-size: 1.85rem; color: #ffffff; }}
    .hero p {{ margin: 0 0 14px 0; font-size: 1rem; color: #ecfeff; }}
    .hero .pill {{ display: inline-block; margin: 0 8px 6px 0; padding: 3px 12px; border-radius: 999px;
                   background: rgba(255,255,255,0.18); font-size: 0.8rem; color: #ffffff; }}
    .notice {{ background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.45);
               padding: 10px 14px; border-radius: 12px; font-size: 0.85rem; margin-top: 18px; }}
    div.stButton > button {{ border-radius: 12px; }}
    </style>
    <div class="hero">
      <h1>AfriHealth Knowledge QA Assistant</h1>
      <p>Ask about a medical condition and get verified answers from NIH sources, with the matched question shown for transparency.</p>
      <span class="pill">{len(kb):,} verified answers</span>
      <span class="pill">Fine-tuned Transformer</span>
      <span class="pill">No generated text</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "reply": {"kind": "message", "text": WELCOME}}]


def use_example(text):
    st.session_state.pending = text


def clear_chat():
    st.session_state.messages = [{"role": "assistant", "reply": {"kind": "message", "text": WELCOME}}]


with st.sidebar:
    st.subheader("About")
    st.write("Answers come from NIH health pages (MedQuAD). A fine-tuned Transformer finds the best matching "
             "stored answer. Nothing is generated, so nothing is made up.")
    st.button("Clear conversation", on_click=clear_chat, use_container_width=True)
    with st.expander("Technical info"):
        st.write(f"**Model:** {MODEL_SOURCE}")
        st.write(f"**Knowledge base:** {len(kb):,} NIH answers")
        st.write(f"**Similarity threshold:** {THRESH:.3f}")
        memory = peak_memory_mb()
        if memory:
            st.write(f"**Peak memory:** {memory} MB")

# the chat box stays at the bottom of the page; answer a new question before drawing the history
typed = st.chat_input("Ask about a condition, e.g. What are the symptoms of Glaucoma?")
question = st.session_state.pop("pending", None) or typed

if question:
    with st.spinner("Searching the knowledge base..."):
        reply = build_reply(question)
    st.session_state.messages.append({"role": "user", "text": question})
    st.session_state.messages.append({"role": "assistant", "reply": reply})

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["text"])
    else:
        with st.chat_message("assistant", avatar="🩺"):
            render_reply(message["reply"])

# example questions, shown until the first question is asked
if len(st.session_state.messages) == 1:
    st.caption("Try an example")
    columns = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        columns[i % 2].button(example, key=f"example_{i}", on_click=use_example, args=(example,),
                              use_container_width=True)

st.markdown(
    '<div class="notice">This assistant gives general health information only. It does not diagnose, '
    'and it is not a substitute for a doctor. In an emergency, contact your local emergency service.</div>',
    unsafe_allow_html=True,
)