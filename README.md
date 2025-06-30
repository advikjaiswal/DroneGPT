# Drone Action GPT

## Overview
A project to predict drone actions using a GPT model, with a Flask backend and a React frontend.

## Setup

### 1. Clone the repository
```sh
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### 2. Python Backend

- Create and activate a virtual environment:
  ```sh
  python3 -m venv venv
  source venv/bin/activate
  ```
- Install dependencies:
  ```sh
  python3 -m pip install -r requirements.txt
  ```
- Start the backend:
  ```sh
  python3 api.py
  ```
  The backend runs on `http://localhost:5001`.

### 3. Frontend

- Go to your frontend directory (if separate):
  ```sh
  cd frontend
  npm install
  npm run dev
  ```
  The frontend runs on `http://localhost:5173`.

### 4. Connecting Frontend and Backend

- The frontend should send API requests to `http://localhost:5001`.
- CORS is enabled in the backend for local development.

## Notes

- Update API URLs in the frontend if you change backend ports.
- For production, use a production WSGI server for Flask. 