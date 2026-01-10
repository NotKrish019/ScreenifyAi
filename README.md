# 🚀 AI Resume Screening System

A sophisticated AI-powered resume screening system that analyzes and ranks candidates based on job description matching using NLP techniques.

## ✨ Features

- 📄 **Multi-format Support**: PDF, DOCX, DOC, TXT
- 🧠 **Advanced NLP**: Tokenization, lemmatization, named entity recognition
- 🎯 **Smart Matching**: TF-IDF vectorization with cosine similarity
- 💼 **Skill Detection**: Identifies and matches technical and soft skills
- 📊 **Intelligent Ranking**: Multi-factor scoring with detailed breakdowns
- 🌐 **Modern UI**: Clean, responsive interface
- ⚡ **Fast API**: RESTful backend with automatic documentation

## 🛠️ Tech Stack

**Backend:**
- Python 3.9+
- FastAPI
- scikit-learn
- NLTK
- pdfplumber/PyPDF2
- python-docx

**Frontend:**
- HTML5
- CSS3
- Vanilla JavaScript

## 📦 Installation & Setup

### First Time Setup (For Both You and Your Friend)

1. **Clone the repository:**
```bash
git clone https://github.com/Neemayg/AI-Resume-Screening-System.git
cd AI-Resume-Screening-System
```

2. **Run the setup script:**
```bash
chmod +x setup.sh
./setup.sh
```

That's it! The script will:
- ✅ Check Python installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Set up the project structure

### Running the Application

**Option 1: Quick Start (Recommended) ⚡**
```bash
./start.sh
```

**Option 2: Manual Start**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in your browser.

## 🔄 For Your Friend - Syncing Latest Changes

When you push new code, your friend just needs to run:

```bash
./sync.sh
```

This will automatically:
1. 📥 Pull the latest code
2. 📦 Update dependencies
3. ✅ Ensure everything is ready to run

Then start the server:
```bash
./start.sh
```

## 🎯 Quick Commands Cheat Sheet

| Task | Command |
|------|---------|
| First time setup | `./setup.sh` |
| Start server | `./start.sh` |
| Sync latest code | `./sync.sh` |
| Manual server start | `cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8000` |

## 📚 API Documentation

Once the server is running, visit:
- **API Docs:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

### Main Endpoints

- `POST /upload` - Upload and analyze resume
- `POST /analyze` - Analyze with job description
- `GET /jobs` - Get available job roles
- `GET /skills` - Get skills database
- `GET /candidates` - Get all analyzed candidates

## 🐛 Troubleshooting

### "No module named uvicorn"
Run the setup script again:
```bash
./setup.sh
```

### "Could not import module 'app'"
Make sure you're using the correct command. The app is in `main.py`:
```bash
uvicorn main:app --reload --port 8000
```

### Port already in use
Kill the process using port 8000:
```bash
lsof -ti:8000 | xargs kill -9
```

### Python version issues
Make sure you have Python 3.9 or higher:
```bash
python3 --version
```

## 📁 Project Structure

```
AI-Resume-Screening-System/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── nlp_engine.py        # NLP processing
│   ├── resume_parser.py     # Resume parsing
│   ├── ranking.py           # Candidate ranking
│   ├── requirements.txt     # Python dependencies
│   └── uploads/             # Uploaded resumes
├── frontend/
│   └── index.html           # Web interface
├── sample_resumes/          # Example resumes
├── setup.sh                 # First-time setup script
├── start.sh                 # Quick start script
└── sync.sh                  # Sync & update script
```

## 🤝 Contributing

This is a private project. For collaborators:

1. **Pull latest changes:**
   ```bash
   ./sync.sh
   ```

2. **Make your changes**

3. **Push changes:**
   ```bash
   git add .
   git commit -m "Your message"
   git push origin main
   ```

## 📝 License

Private project - All rights reserved.

## 👥 Team

This project is maintained by the development team.

---

**Made with ❤️ using Python & FastAPI**