# Healthcare QA Assistant (web app)

English question-answering assistant for health information, built on MedQuAD (NIH sources).
A fine-tuned Transformer (bi-encoder) finds the stored NIH answer that best matches the question.
It returns existing verified answers, so it does not invent medical text.

**Limits:** it answers questions that **name a condition**. It is not a symptom checker, and it gives general
information only, not a diagnosis or medical advice.

## Files in this folder

| File | Purpose |
|---|---|
| `streamlit_app.py` | Streamlit app (the deployed version) |
| `app.py` | Gradio version (optional, for running locally) |
| `requirements.txt` | Python packages |
| `config.json`, `kb.csv`, `kb_embeddings.npy` | Exported by Notebook 04 (`app_bundle.zip`) |
| `model/` (optional) | Local copy of `models/final`. If absent, the app downloads the model from the Hugging Face Hub repo named in `config.json` |

`.streamlit/config.toml` (colors) sits in the **repository root**, not here.

## Run on your computer

From the **repository root** (so the theme file is found):

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Gradio version (optional): `pip install "gradio>=6,<7"` then `cd app && python app.py`.

The app looks for the model in this order:
1. the folder in the `MODEL_PATH` environment variable
2. the `model/` folder next to the app file (it must contain `modules.json`)
3. the Hugging Face repo named in `config.json` (`model_hub_repo`)

## Deploy on Streamlit Community Cloud (free)

1. Push the repository to GitHub (public). Do not ignore `kb_embeddings.npy`.
2. Go to https://share.streamlit.io, sign in with GitHub, and choose **Create app**.
3. Pick the repository and branch, and set **Main file path** to `app/streamlit_app.py`.
4. Under **Advanced settings** choose Python 3.12, then **Deploy**.
5. The first start downloads the model (about 440 MB), so wait a few minutes. Use **Manage app** to read the logs.

The model repository on Hugging Face must be public. Community Cloud apps sleep after 12 hours without traffic,
so open the link before showing it.

## Settings in `config.json`

- `threshold`: minimum similarity to answer. Below it the app says the question is out of scope.
- `min_name_overlap` (optional, default 0.5): how much of the matched condition name must appear in the question.
  Lower it if valid questions get refused, raise it if symptom descriptions get answered.
- `prefix`, `max_len`, `model_hub_repo`: must match the trained model.