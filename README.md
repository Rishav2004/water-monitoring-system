🌊 Water Body Monitoring System

An end-to-end system for real-time monitoring and classification of water bodies using Machine Learning, Computer Vision, and Web Technologies.
The system analyzes images/videos of water and classifies them into different categories like clean, polluted, algae presence, oil spills, etc.

🚀 Features
📸 Image-based water quality detection
🤖 Machine Learning model for classification
🌐 Web-based dashboard (Frontend + Backend)
⚡ Real-time API for predictions
🧪 Supports multiple water categories:
Clean Water
Polluted Water
Algae Present
Oil Spill Water
Foam Water

🛠️ Tech Stack
🔹 Frontend
HTML, CSS, JavaScript / React

🔹 Backend
Node.js
Express.js

🔹 Machine Learning
Python
Flask API
OpenCV
Scikit-learn
NumPy

📂 Project Structure
water-monitoring-system/
│
├── frontend/        # UI (React / HTML)
├── backend/         # Node.js server
├── ml_model/        # ML model + Flask API
│   ├── flask_api.py
│   ├── requirements.txt
│   └── venv/
│
└── README.md

⚙️ Setup & Installation
🔹 1. Clone the Repository
git clone <your-repo-link>
cd water-monitoring-system

🌐 Run Frontend
cd frontend
npm install
npm run dev

🖥️ Run Backend
cd backend
npm install
npm run dev

🤖 Run ML Model
Step 1: Go to ML folder
cd ml_model

Step 2: Create Virtual Environment (Python 3.10 recommended)
py -3.10 -m venv venv

Step 3: Activate
.\venv\Scripts\Activate.ps1

Step 4: Install dependencies
pip install -r requirements.txt

Step 5: Run API
python flask_api.py

🌍 API Endpoint
http://127.0.0.1:5001

🧪 How It Works
User uploads image via frontend
Frontend sends request to backend
Backend forwards image to ML API
ML model processes image using OpenCV
Prediction returned and displayed
⚠️ Important Notes

Use Python 3.10 or 3.11 (avoid 3.14 due to compatibility issues)
Ensure all 3 services are running:
Frontend
Backend
ML API
Model warnings (sklearn version mismatch) can be ignored if output is correct
📸 Future Improvements

🎥 Live video stream monitoring
☁️ Cloud deployment (AWS / Azure)
📊 Advanced analytics dashboard
📍 Geo-tagging of water bodies

👨‍💻 Author

Rishav

📜 License

This project is for educational purposes.

⭐ If you like this project

Give it a ⭐ on GitHub!
