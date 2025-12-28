import numpy as np

def solve_iterative(initial_board, model_manager):
    """
    Exact logic from 'sudoku_solve' in your notebook.
    Fills cells with highest probability iteratively.
    """
    puzzle = np.array(initial_board).copy()
    iter_count = 0
    max_iter = 82 # Safety break
    
    # While there are zeros (empty cells)
    while 0 in puzzle and iter_count < max_iter:
        
        # Normalize Input: (puzzle/9) - 0.5
        input_norm = (puzzle / 9.0) - 0.5
        input_puzz = input_norm.reshape(1, 9, 9, 1)
        
        # Predict
        output_puzz = model_manager.predict_solver(input_puzz)
        
        # Get Predictions and Probabilities
        # Note: Your code adds +1 because argmax(0) is digit 1
        out = np.argmax(output_puzz, axis=-1).squeeze() + 1 
        maxp = np.max(output_puzz, axis=-1).squeeze()
        
        # Mask already filled positions (set prob to -1)
        maxp[puzzle != 0] = -1.0
        
        # Find index of maximum probability in the entire grid
        flat_idx = np.argmax(maxp)
        row, col = divmod(flat_idx, 9)
        
        # If the best probability is very low, we might be stuck
        if maxp[row, col] == -1: 
            break 
            
        # Fill the cell
        puzzle[row, col] = out[row, col]
        iter_count += 1
        
    return puzzle.tolist()