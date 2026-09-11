# Interview Forge — Deployment Checklist

## GitHub
- [ ] Create `interview-forge` repository
- [ ] Upload all project files
- [ ] Confirm `.streamlit/secrets.toml` is NOT uploaded
- [ ] Confirm no API key is present in source code

## Groq
- [ ] Create Groq account/API key
- [ ] Use a currently supported free-tier model
- [ ] Recommended default: `openai/gpt-oss-20b`

## Streamlit Community Cloud
- [ ] Connect GitHub
- [ ] Select repository
- [ ] Branch: `main`
- [ ] Main file: `app.py`
- [ ] Python: 3.12
- [ ] Add secrets:

```toml
GROQ_API_KEY = "YOUR_KEY"
GROQ_MODEL = "openai/gpt-oss-20b"
```

- [ ] Deploy
- [ ] Upload a text-based PDF CV
- [ ] Optionally upload a text-based job description
- [ ] Click `Forge Interview Plan`

## If PDF extraction returns little/no text
The MVP uses `pypdf`, which works best with selectable-text PDFs. Scanned/image-only PDFs need an OCR layer in a future version.
