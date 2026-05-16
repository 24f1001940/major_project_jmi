# Vision-Based Queue Analytics

Prepared by **MOHD SAQIB**.

This project contains a YOLOv8 and OpenCV queue analytics application with a Streamlit dashboard for local use, plus a lightweight static page for Vercel deployment.

## Local App

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

The local app uses `models/best.pt` and the sample video in `dataset_videos/`. Runtime uploads and generated output are ignored by git.

## Vercel Deployment

Vercel serves `index.html` as a static project page. The heavy local runtime files are excluded through `.vercelignore` because Vercel is not suited for the OpenCV processing window or long-running Streamlit processor.

```powershell
vercel --prod
```
