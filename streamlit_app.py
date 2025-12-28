import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt

# Connect to your running FastAPI backend
API_URL = "http://localhost:8000/solve"

st.set_page_config(page_title="Sudoku AI Solver", layout="wide")

st.title("🧩 Neural Sudoku Solver")
st.markdown("""
Upload an image of a Sudoku puzzle. The AI will:
1. **Extract** the grid using OpenCV.
2. **Recognize** digits using a CNN trained on SVHN.
3. **Solve** the puzzle using an Iterative Neural Network.
""")

def plot_grid(grid, title):
    """
    Draws a professional Sudoku grid using Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Draw the grid lines
    for i in range(10):
        # Thick lines for 3x3 blocks, thin for cells
        linewidth = 2.5 if i % 3 == 0 else 0.5
        ax.axhline(i, color='black', linewidth=linewidth)
        ax.axvline(i, color='black', linewidth=linewidth)

    # Place the numbers
    # Grid is 9x9, but matplotlib plots 0..9. We center text at x+0.5, y+0.5
    # Note: Matplotlib coordinates start from bottom-left by default, 
    # but matrices are top-left. We invert y-axis logic or use matshow.
    # Simpler approach: Iterate and text.
    
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0) # Invert y to match matrix convention (row 0 at top)

    for row in range(9):
        for col in range(9):
            val = grid[row][col]
            if val != 0:
                ax.text(col + 0.5, row + 0.5, str(val), 
                        ha='center', va='center', fontsize=14, color='#333333')

    ax.axis('off')
    ax.set_title(title, y=-0.1) # Title at bottom
    return fig

# --- Sidebar ---
st.sidebar.header("Input Image")
uploaded_file = st.sidebar.file_uploader("Choose a Sudoku Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    st.sidebar.image(uploaded_file, caption="Original Image", use_column_width=True)
    
    if st.button("🧩 Solve Puzzle"):
        with st.spinner("Processing... (This might take a moment)"):
            try:
                # Prepare file for API
                files = {"file": uploaded_file.getvalue()}
                
                # Call the Backend
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Get the boards from the JSON response
                    # Note: These keys match what we defined in app.py
                    detected_board = data["detected_board"]
                    solved_board = data["solution"]
                    
                    # Layout: 2 Columns for Before/After
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.pyplot(plot_grid(detected_board, "Detected Digits (Model 1)"))
                        st.info("These are the numbers the AI 'saw' in your image.")

                    with col2:
                        st.pyplot(plot_grid(solved_board, "AI Solution (Model 2)"))
                        st.success("Here is the completed puzzle!")

                else:
                    st.error(f"Server Error: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the backend. Is 'uvicorn app:app' running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")