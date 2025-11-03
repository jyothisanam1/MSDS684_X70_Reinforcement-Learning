# Dynamic Programming for Solving Finite MDPs

This project implements and analyzes two foundational Dynamic Programming (DP) algorithms for finding optimal policies in finite Markov Decision Processes (MDPs): **Policy Iteration** and **Value Iteration**. The analysis compares their performance, convergence properties, and the impact of implementation details like synchronous vs. in-place updates.

The experiments are conducted on a custom-built GridWorld environment and the standard `FrozenLake-v1` environment from Gymnasium.

## File Structure

*   `custom_gridworld.py`: A Python module containing the `CustomGridWorld` environment class and the core implementations of the four DP algorithms (Policy Iteration Sync/In-Place, Value Iteration Sync/In-Place) and visualization functions.
*   `analysis.ipynb`: The main Jupyter Notebook used to run all experiments, compare the algorithms, generate visualizations, and produce the final analysis tables.
*   `requirements.txt`: A list of all necessary Python packages to run the project.

## Prerequisites

*   Python 3.8 or higher
*   `pip` and `venv` (usually included with Python)

## Setup and Installation

Follow these steps to set up a virtual environment and install the required dependencies. This ensures that the project's packages do not interfere with your system's global Python installation.

**1. Navigate to the Project Directory**

Open your terminal or command prompt and navigate to the root folder of this project.

```bash
cd path/to/your/project-folder
```

**2. Create a Python Virtual Environment**

Create a virtual environment named `venv`.

```bash
python3 -m venv venv
```

**3. Activate the Virtual Environment**

You must activate the environment before installing packages. The command differs for Windows and macOS/Linux.

*   **On Windows:**
    ```bash
    .\venv\Scripts\activate
    ```

*   **On macOS and Linux:**
    ```bash
    source venv/bin/activate
    ```

After activation, you should see `(venv)` at the beginning of your terminal prompt.

**4. Install Required Packages**

Use `pip` to install all the libraries listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

**5. Link the Virtual Environment to Jupyter**

This crucial step makes your virtual environment available as a kernel within Jupyter Notebook, ensuring you are using the correct packages.

```bash
python -m ipykernel install --user --name=dp-lab-env
```
*(You can replace `dp-lab-env` with any name you prefer)*

## How to Run the Project

**1. Launch Jupyter Notebook**

Make sure your virtual environment is still active (you see `(venv)` in your prompt) and run the following command:

```bash
jupyter notebook
```

This will open a new tab in your web browser with the Jupyter interface.

**2. Open the Analysis Notebook**

In the browser tab, navigate to and click on the `analysis.ipynb` file to open it.

**3. Select the Correct Kernel**

Once the notebook is open, verify that you are using the correct kernel.
*   In the top-right corner of the notebook, you should see the kernel name (e.g., "Python 3").
*   Click on it, go to `Kernel > Change kernel`, and select **`dp-lab-env`** (or the name you chose in the setup step).

**4. Run the Cells**

You can now run the cells in the notebook sequentially to execute the experiments, generate the plots, and view the analysis.
*   To run a single cell, select it and press `Shift + Enter`.
*   To run the entire notebook, go to `Kernel > Restart & Run All`.