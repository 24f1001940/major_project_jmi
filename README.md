# Vision-Based Queue Analytics

Prepared by **MOHD SAQIB**.

This project contains a YOLOv8 and OpenCV queue analytics application with a Streamlit dashboard for local use, plus a browser-based Vercel deployment that can process uploaded videos directly on the client.

## Local App

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

The local app uses `models/best.pt` and the sample video in `dataset_videos/`. Runtime uploads and generated output are ignored by git.

## Vercel Deployment

Vercel serves `index.html` as a static browser app. It supports video upload, ROI selection, live person detection, simple tracking, queue metrics, charts, event logs, and CSV export in the browser. The heavy local runtime files are excluded through `.vercelignore` because Vercel is not suited for the OpenCV processing window or long-running Streamlit processor.

```powershell
vercel --prod
```
