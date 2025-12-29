
# 🧩 Neural Sudoku Solver

An AI-powered web application that solves Sudoku puzzles from images using Computer Vision and Deep Learning. 

Unlike traditional backtracking algorithms, this project uses a **Pure Neural Network approach** to solve the board iteratively, mimicking human intuition.


## 🚀 Features

* **Computer Vision Pipeline**: Automatically detects, crops, and processes Sudoku grids from raw images using OpenCV.
* **Digit Recognition (CNN)**: Recognizes handwritten or printed digits (1-9) using a CNN trained on the SVHN (Street View House Numbers) dataset.
* **Neural Solver**: A specialized Convolutional Neural Network that solves the puzzle iteratively by filling the "most confident" cell one at a time, rather than using recursion.
* **Full-Stack Interface**: 
    * **Backend**: FastAPI for high-performance inference.
    * **Frontend**: Streamlit for an interactive drag-and-drop UI.

---

## 📂 Project Structure

```text
sudoku-deploy/
├── models/                  # Place trained .hdf5 models here
├── images/                  # Assets (e.g., zero_template.jpg)
├── src/                     # Core Logic
│   ├── image_processing.py  # OpenCV grid extraction & warping
│   ├── neural_solver.py     # Iterative neural solving logic
│   └── model_loader.py      # TensorFlow model management
├── app.py                   # FastAPI Backend
├── ui.py                    # Streamlit Frontend
├── Dockerfile               # Container config
└── requirements.txt         # Dependencies
```

🛠️ Installation & Setup
------------------------

### Prerequisites

1.  **Download Models**: You must download the trained models and zero_template.jpg and place them in the models/ and images/ directory respectively:
    
    *   digit\_svhn-196-0.14.hdf5
        
    *   sudoku\_conv-20-0.10.hdf5
        
2.  **Assets**: Ensure images/zero\_template.jpg exists.
    

### Option A: Run Locally (Standard)

 ```bash
pip install -r requirements.txt
 ```
    
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
    
```bash
streamlit run ui.py
```

Access the app at http://localhost:8501
    
### Option B: Run with Docker (Recommended)

```bash
docker-compose up --build
```
    
    *   Frontend: http://localhost:8501
        
    *   API Docs: http://localhost:8000/docs
        

🧠 How It Works
---------------

1.  **Preprocessing**: The input image is thresholded and warped to a flat 9x9 grid using cv2.getPerspectiveTransform.
    
2.  **Cell Extraction**: The grid is sliced into 81 individual 32x32 pixel images.
    
3.  **Digit Classification**: A CNN model processes the 81 cells to identify existing numbers.
    
4.  **Iterative Solving**:
    
    *   The board state is fed into the Solver Network.
        
    *   The network predicts probabilities for all empty cells.
        
    *   The cell with the **highest confidence** is filled.
        
    *   This repeats until the board is full.
        

📄 License
----------

This project is open-source. 