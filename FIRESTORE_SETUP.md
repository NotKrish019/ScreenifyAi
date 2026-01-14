# Setting up History Feature with Firebase Firestore

To enable the "History" feature where you can revisit past analysis results, you need to connect the application to Google Firebase.

## Prerequisites
1. A Google Account.

## Steps

### 1. Create a Firebase Project
1. Go to [Firebase Console](https://console.firebase.google.com/).
2. Click **Add project** and follow the setup steps (you can disable Google Analytics).
3. Once created, click on **Build** in the left sidebar, then select **Firestore Database**.
4. Click **Create Database**.
   - Select **Start in production mode** (or test mode if you prefer).
   - Choose a location (e.g., `us-central1`).

### 2. Generate Service Account Key
1. In the Firebase Console, click the **Gear icon** (Project Settings) next to "Project Overview".
2. Go to the **Service accounts** tab.
3. Click **Generate new private key**.
4. This will download a JSON file containing your credentials.

### 3. Place the Key
1. Rename the downloaded file to `serviceAccountKey.json`.
2. Move this file into the `backend/` directory of this project.
   - Path: `backend/serviceAccountKey.json`

### 4. Restart the Backend
1. If your backend is running, stop it (Ctrl+C).
2. Start it again using `start.sh` or `python backend/main.py`.

## Verification
- When the backend starts, you should see a log message: `Firestore initialized successfully.`
- In the frontend, click the **History** button. You should see your past analyses appear there after you run them.
