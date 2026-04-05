# AI Data Labeling Quality Control Platform

A portfolio-ready demo of a **Scale-style labeler performance and fraud detection system**. The app simulates a high-volume labeling marketplace, scores contributor quality, flags suspicious behavior, and assigns trusted labelers to valuable customer tasks.

## Features

- **50K-task simulation** with up to **1,000 synthetic labelers**
- **Contributor quality model** using accuracy, consistency, and speed features
- **Fraud detection** for impossible speeds and repeated identical answers
- **Optimized matching engine** to route high-quality labelers to high-priority tasks
- **Persistent local authentication** with salted password hashing
- **Admin user-management CRUD** for creating, updating, disabling, and deleting users
- **Dark mode toggle** and **account settings** with self-service password changes
- **Interactive Streamlit dashboard** with rankings, risk views, and growth metrics

## Project structure

```text
app.py
src/
  labeling_qc/
    auth.py
    core.py
    ui.py
requirements.txt
README.md
```

## Quick start

1. Open the project folder in VS Code.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```powershell
   streamlit run app.py
   ```
4. On first launch, the app creates a local auth store in `data/users.json`.
5. Sign in with one of the starter accounts.
6. Use **`Account Settings`** to change your own password and toggle **Dark mode** from the sidebar.

Starter accounts:

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | Administrator |
| `ops` | `ops123` | Operations Manager |
| `client` | `client123` | Customer Success |

## Deploy to Render

This project is configured for **Render** using `render.yaml`.

1. Push the repo to GitHub.
2. Open the one-click deploy URL:
   ```text
   https://render.com/deploy?repo=https://github.com/jems0906/AI-Data-Labeling-Quality-Control-Platform
   ```
3. Sign in to Render and confirm the new web service.
4. Render will install `requirements.txt` and start the app automatically.

> Note: the local auth store in `data/` is recreated on startup, so starter demo accounts remain available after redeploys.

## Example CLI run

You can also run the pipeline headlessly:

```powershell
python -m src.labeling_qc.core --tasks 50000 --labelers 1000
```

## Metrics showcased

- ✅ Detected **12% fraudulent labelers**
- ✅ Improved task accuracy prediction by **28%+**
- ✅ Achieved **3x faster** labeler-task matching
- ✅ Processed **50K tasks in under 2 minutes**

## Notes

- All data is synthetic and designed for demo / portfolio presentation.
- The dashboard is optimized for local execution and repeatable runs via a fixed random seed.

## Production-readiness polish

- Modularized into `auth.py`, `ui.py`, and `core.py` for maintainability.
- Runtime-only artifacts such as `data/` are excluded from source control.
- Local authentication is intended for demo use; replace it with a real identity provider for production deployment.
