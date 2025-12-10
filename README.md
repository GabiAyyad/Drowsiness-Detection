# Driver Drowsiness Detection System

A real‑time driver drowsiness detection system using **Mediapipe Face Mesh**, **Eye Aspect Ratio (EAR)**, **Head Pose Estimation**, and an audio alarm.

## 🚀 Features
- EAR‑based eye‑closure detection  
- Head pitch/yaw monitoring for nodding or side distraction  
- Automatic EAR calibration on start  
- Alarm triggers when drowsiness is confirmed  
- Works with any laptop or USB camera  
- Clean project structure and easy installation  

## 📂 Project Structure
```
Drowsiness-Detection/
│── src/
│   ├── Final_prototype.py
│   ├── Cam_test.py
│   ├── Sound_Testing.py
│
│── assets/
│   ├── alarm1.mp3
│
│── requirements.txt
│── README.md
```

## 🛠 Installation
```
pip install -r requirements.txt
```

## ▶️ Run the main program
```
python src/Final_prototype.py
```

## 📝 Notes
- If the camera doesn’t open, try switching the camera index inside the code.
- Make sure *Alarm.mp3* stays inside the **assets/** folder.

## 📸 Demo (optional)


---

Made by Gabriel Ayyad  
