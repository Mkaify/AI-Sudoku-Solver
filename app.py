from fastapi import FastAPI, UploadFile, File, HTTPException
from src.model_loader import ModelManager
from src.image_processing import process_pipeline
from src.neural_solver import solve_iterative
import numpy as np
import uvicorn

app = FastAPI()
manager = ModelManager()

@app.on_event("startup")
def startup():
    manager.load_models()

@app.post("/solve")
async def solve(file: UploadFile = File(...)):
    if not manager.digit_model or not manager.solver_model:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        # 1. Image -> 81 blocks (Batch of 32x32)
        image_data = await file.read()
        blocks = process_pipeline(image_data) # Returns (81, 32, 32, 1)

        # 2. Recognize Digits
        preds = manager.predict_digits(blocks) # Returns (81, 10)
        
        # Convert predictions to 9x9 board
        # Logic: argmax gives 0..9. 
        # Your model output has 10 classes (0=Empty, 1..9=Digits) or (0..9 digits)?
        # Looking at your code (Line 220): Dense(10). 
        # Line 229: puzzle = np.argmax(out, axis=-1).reshape(9,9)
        # This implies class 0 is empty/zero, and 1-9 are digits.
        initial_board = np.argmax(preds, axis=-1).reshape(9, 9)

        # 3. Solve using Iterative Neural approach
        solved_board = solve_iterative(initial_board, manager)

        return {
            "detected_board": initial_board.tolist(),
            "solution": solved_board
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)