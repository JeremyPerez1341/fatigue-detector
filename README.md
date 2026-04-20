# 🚗 Detector de Fatiga y Distracción (Prototipo)

Prototipo de sistema de visión computacional para detectar signos de fatiga y distracción en conductores usando la cámara.

## 🧠 Tecnologías usadas

* Python 3.10.11
* MediaPipe 0.10.9
* OpenCV
* NumPy

---

## ⚙️ Requisitos

Antes de empezar, asegúrate de tener instalado:

* Python 3.10.11
  (IMPORTANTE: no usar Python 3.14 por incompatibilidad con MediaPipe)

---

## 🚀 Instalación paso a paso

### 1. Clonar repositorio

```bash
git clone <URL_DEL_REPO>
cd fatigue-detector
```

---

### 2. Crear entorno virtual

```bash
py -3.10 -m venv venv
```

---

### 3. Activar entorno virtual

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

---

### 4. Actualizar pip

```bash
python -m pip install --upgrade pip
```

---

### 5. Instalar dependencias (versiones fijas)

```bash
pip install mediapipe==0.10.9 opencv-python numpy
```

---

## ▶️ Ejecución

```bash
python main.py
```

---

## 📌 Resultado esperado

* Se abrirá la cámara
* Se detectará el rostro
* Se dibujarán puntos faciales en tiempo real

---

## ⚠️ Problemas comunes

### ❌ Error: `mediapipe has no attribute 'solutions'`

* Solución: usar `mediapipe==0.10.9`

### ❌ Error con NumPy (`_multiarray_umath`)

* Solución: eliminar carpeta `venv` y recrear entorno

### ❌ No abre la cámara

* Cambiar:

```python
cv2.VideoCapture(0)
```

por:

```python
cv2.VideoCapture(1)
```

---

## 🧪 Estado del proyecto

✔ Detección facial en tiempo real
⬜ Detección de ojos (pendiente)
⬜ Detección de fatiga
⬜ Detección de distracción

---

## 📌 Próximos pasos

* Implementar detección de ojos (EAR)
* Detectar fatiga por tiempo de cierre ocular
* Detectar distracción (orientación de cabeza)

---
