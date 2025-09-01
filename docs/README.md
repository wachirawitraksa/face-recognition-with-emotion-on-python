# Face Recognition System with Emotion Detection
ระบบตรวจจับและระบุใบหน้าแบบ Real-time พร้อมระบุชื่อและอารมณ์ (Windows / Python 3.11)

> Author: Wachirawit Raksa · Modified: 2025  
> Stack: OpenCV + MediaPipe + face_recognition (Dlib) + DeepFace (TensorFlow)

---

## ✨ ฟีเจอร์
- ตรวจจับใบหน้าแบบเรียลไทม์ด้วย **MediaPipe Face Detection**
- ระบุ “ชื่อบุคคล” ด้วย **face_recognition (Dlib encodings)**
- วิเคราะห์อารมณ์ (emotion) ด้วย **DeepFace**
- แคช/ทำ smoothing อารมณ์เพื่อให้ผลนิ่งขึ้น
- บันทึก/โหลดฐานข้อมูลใบหน้าเป็นไฟล์ **PKL** เพื่อบูทเร็ว
- ปรับแต่งพารามิเตอร์ FPS / ความละเอียด / tolerance ได้

---

## 🧱 โครงสร้างหลักของโปรเจค
```text
face-recognition-with-emotion-on-python/
├─ dataset/                  # โฟลเดอร์รูปภาพใบหน้าที่จัดตามชื่อคน
│  ├─ person1/ img001.jpg ...
│  └─ person2/   img010.jpg ...
├─ docs/                  # โฟลเดอร์รูปภาพใบหน้าที่จัดตามชื่อคน
│  ├─ flowchart.png
│  └─ README.md
├─ src/
│  └─ haarcascade_frontalface_default.xml       # Object Detection Algorithm
├─ trained/
│  └─ face_dataset.pkl       # ไฟล์เข้ารหัสใบหน้า (สร้างอัตโนมัติเมื่อโหลดสำเร็จ)
├─ main.py                   # สคริปต์หลัก
└─ requirements.txt          # รายการไลบรารีเวอร์ชันที่ล็อกไว้
```

### โครงสร้าง `dataset/` (ตัวอย่าง)
```text
dataset/
├─ person1/
│  ├─ 01.jpg
│  ├─ 02.jpg
│  └─ 03.jpg
└─ person2/
   ├─ a.jpg
   └─ b.png
```
- รองรับ `.jpg/.jpeg/.png`  
- ควรมี 3–10 รูปต่อคนในหลายมุม/หลายสภาพแสง

---

## 🖥️ ข้อกำหนดระบบ
- Windows 10/11  
- **Python 3.11.9**  
- กล้องเว็บแคม  
- (เลือกได้) GPU ที่รองรับ TensorFlow  

---

## 📦 การติดตั้ง
```powershell
# ไปยังโฟลเดอร์โปรเจค
cd C:\Users\<username>\OneDrive\Documents\face-recog-emotion

# สร้างและเปิดใช้งาน venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# อัป pip
python -m pip install --upgrade pip

# ติดตั้ง dependencies
pip install -r requirements.txt

# ตรวจสอบความเข้ากันได้
pip check
```

---

## 🚀 การใช้งาน
1. เตรียมรูปใน `dataset/`  
2. รัน:
   ```powershell
   python main.py
   ```
3. กด `q` หรือ `ESC` เพื่อออก  

---

## ⚙️ การตั้งค่าหลัก (ใน `FaceRecognitionSystem.__init__`)
- `FACE_RECOGNITION_TOLERANCE` (default 0.55)  
- `MIN_FACE_SIZE` – ขนาดกรอบหน้าขั้นต่ำ  
- `EMOTION_ANALYSIS_INTERVAL` – ความถี่การวิเคราะห์อารมณ์  
- `PROCESSING_SCALE` – ย่อภาพเพื่อตรวจจับ (ลดลง = เร็วขึ้น)  
- `TARGET_FPS` – ตั้งค่า FPS  
- `EMOTION_SMOOTHING_FRAMES` – จำนวนเฟรมสำหรับ smoothing  

---

## 🧪 เคล็ดลับคุณภาพ
- รูปหลายมุม/หลายแสงต่อคน  
- ลด `PROCESSING_SCALE` (เช่น 0.5) หากเครื่องช้า  
- เปิด MJPG ในกล้อง → FPS ดีขึ้น  

---

## 🧯 Troubleshooting
### MediaPipe / Protobuf Error
```
RuntimeError: Failed to parse: node {...}
```
แก้โดย:
```powershell
pip uninstall -y protobuf numpy
pip install protobuf==4.25.8 numpy==1.26.4
pip install mediapipe==0.10.21
pip check
```

### OpenCV เตือน numpy
```powershell
pip install opencv-python==4.11.0.86 opencv-contrib-python==4.11.0.86
```

---

## 🔏 Privacy
- ประมวลผลบนเครื่อง (on-device)  
- PKL เก็บเฉพาะ encodings + ชื่อ ไม่เก็บรูปจริง  

---

## 📜 Author
Wachirawit Raksa
(27/07/2025)

---

## 🙏 Credits
- MediaPipe by Google Research  
- dlib / face_recognition by Davis E. King  
- DeepFace by Serengil & contributors  
- OpenCV community  
