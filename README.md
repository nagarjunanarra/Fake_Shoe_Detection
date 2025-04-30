# 👟 Fake Shoes Review - AI-Based Shoe Authenticity Detection

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![HTML](https://img.shields.io/badge/Frontend-HTML%2FCSS-blue?logo=html5)

![Web App Preview](./Frontend/images/webapp_screenshot.png)

## 🧾 Summary
*Project Name:* Fake Shoes Review - AI Detection App  
*Role:* Full Stack Developer, AI Model Trainer  
*Contributions:* Designed full HTML/CSS UI, built FastAPI backend, trained and integrated a CNN TensorFlow model, implemented real-time prediction with drag-and-drop upload UI.

### 🔍 Sprint Objectives Summary Table
| Sprint | Objective | Outcome |
|--------|-----------|---------|
| 1 | MVP UI with upload flow | ✅ Completed responsive layout + modal design |
| 2 | Model training & pipeline | ✅ Built & trained TensorFlow CNN on shoe dataset |
| 3 | Backend API integration | ✅ FastAPI /upload route with real-time feedback |


### 📌 User Story & Task Summary
- 👟 As a user, I want to upload shoe images and get a real/fake prediction.
- ✅ Built user-friendly upload UI (HTML/CSS + JS logic)
- ✅ Connected backend API to serve ML predictions
- ✅ Integrated modal preview and confidence feedback
- 🧪 Tested model with varied examples

### 🔗 Evidence of Work
- 📂 Public Repo: [GitHub - Fake_Shoe_Detection](https://github.com/nagarjunanarra/Fake_Shoe_Detection)
- ✅ Commit History: [View Commits](https://github.com/nagarjunanarra/Fake_Shoe_Detection/commits/main)
- 🔍 Key Code: [backend/model.py](backend/model.py), [frontend/index.html](frontend/index.html)

```

## ✨ Project Overview
An AI-powered platform to detect *fake shoe reviews and images*. Users upload images and get real-time predictions using a trained CNN model built with TensorFlow. The system uses:

- 🧠 AI-based image classification
- ⚡ FastAPI backend
- 🎨 A clean HTML5/CSS3 frontend

```

## 🧠 AI Model
- Built using TensorFlow and Keras
- Input image size: 150x150
- Trained on real vs fake shoe categories:
  - Fake Air Force
  - Fake Jordan
  - Ori Jordan
  - Other Class
- Accuracy ~90%+ on validation set

Model pipeline:


[Upload Image] → [Preprocess] → [CNN] → [Softmax Prediction] → [Class Label + Confidence]


```

## 🖼️ Frontend UI
HTML + CSS layout with:
- Hero banner
- Upload section with drag-and-drop
- Modal preview of predictions
- Review cards for explanation

> File: /frontend/index.html

```

## ⚙️ API Backend
FastAPI server with a /upload endpoint that:
- Accepts images
- Processes them
- Returns predicted class and confidence

> File: /backend/main.py

```

## 📁 Project Structure

FakeReview/
├── frontend/
│   ├── index.html          
│   └── images/            
├── backend/
│   ├── main.py             
│   └── model.py            
├── models/
│   └── trained_model.h5    
└── README.md               


```

## 📦 Installation
### 🖥️ Frontend (Static HTML)
Open index.html in browser.

### 🧪 Backend (FastAPI)
bash
pip install fastapi uvicorn pillow tensorflow
uvicorn backend.main:app --reload


```

## 📬 API Endpoints
### POST /upload/
Upload an image to get prediction.
#### Payload:
multipart/form-data
bash
file=<image>

#### Response:
json
{
  "prediction": "Fake Air Force",
  "confidence": 92.34
}


```

## 🤝 Contributing
1. Fork the repo
2. Create a feature branch: git checkout -b feature/your-feature
3. Commit your changes
4. Open a Pull Request

```


> Built by *Nagarjuna Narra*


