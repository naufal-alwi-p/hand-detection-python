## Panduan Install dan Menjalankan

### Prasyarat

- Python 3.12+ terpasang di sistem.
- Pip tersedia (`python -m pip --version`).

### Instalasi (Windows PowerShell)

1. Buat dan aktifkan virtual environment.

	```powershell
	python -m venv .venv
	```

2. Instal dependensi.

	```powershell
	pip install -r requirements.txt
	```

### Menjalankan Program

Pastikan folder `model/` berisi file model yang dibutuhkan:

- [model/gesture_recognizer.task](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task)
- [model/hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)

Jalankan salah satu skrip berikut:

```powershell
python gesture_recognizer.py
```

atau

```powershell
python slide_swipe.py
```

## Referensi Versi Pengembangan

- Python: 3.12.8
- Pip: 24.3.1
