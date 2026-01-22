# Backend Deployment Guide (Render)

Since Vercel is not suitable for heavy Python/ML backends, we recommend **Render** or **Railway**.
This guide explains how to deploy the backend to **Render**.

## Prerequisites
1. Push your code to a GitHub repository.

## Step 1: Create a Render Service
1. Go to [dashboard.render.com](https://dashboard.render.com/).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.

## Step 2: Configure the Service
- **Name**: `stocks-backend` (example)
- **Runtime**: **Docker** (Important! Do not choose Python directly as we have complex dependencies)
- **Region**: Choose one close to you (e.g., Singapore or Oregon).
- **Branch**: `main`
- **Root Directory**: `web_app/backend` (This is where the Dockerfile is)

## Step 3: Environment Variables
Add the following environment variables if needed (e.g., API keys).
Currently, no critical secrets are mandated by the code, but if you add them later, do it here.

## Step 4: Deploy
- Click **Create Web Service**.
- Render will build the Docker image. This might take 10-15 minutes because it installs heavy libraries like `torch` or `xgboost`.
- Once finished, you will get a URL like `https://stocks-backend.onrender.com`.

## Step 5: Connect Frontend
1. Go to your **Vercel** project dashboard.
2. Go to **Settings** -> **Environment Variables**.
3. Add a new variable:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://stocks-backend.onrender.com/api` (Don't forget the `/api` at the end!)
4. Redeploy the Frontend on Vercel.

## Troubleshooting
- **Build Failures**: Check the logs. If memory runs out, you might need a paid plan or try to optimize `requirements.txt`.
- **Timeouts**: Render free tier spins down after inactivity. The first request might take 50+ seconds.
