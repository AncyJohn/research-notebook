# Notes:

## 3️⃣ The **right role** for Julia in your journey

Think of Julia as a **thinking tool**, not a job tool.

### A very good pattern for *you*

- **Python** → main workflow, notebooks, projects, GitHub
- **Julia** → side experiments, algorithm intuition, performance tests

Example:

- Prototype an idea in Julia
- Re-implement the final version in Python

This actually makes you *stronger*, not distracted.

## Make Anaconda Prompt start in E:

If you prefer it to *always* open on E:, you can change the shortcut.

### 🔹 Steps:

1. Right-click **Anaconda Prompt** → **More → Open file location**
2. Right-click the shortcut → **Properties**
3. In **Start in:** change:
    
    ```
    %USERPROFILE%
    
    ```
    
    to:
    
    ```
    E:\
    
    ```
    
    or:
    
    ```
    E:\projects
    
    ```
    
4. Click **OK**

Next time it opens → straight to E:.

## The idea of Conda environments (intuitively)

Think of Conda like a **warehouse** and environments like **separate rooms**.

- Each environment has:
    - Its own Python version
    - Its own installed packages
- Only **one room is active at a time**

When you activate an environment:

- PATH is adjusted temporarily
- `python`, `pip`, `jupyter` now point to that environment

## What is `base` specifically?

`base` is:

- The **first environment** created when you installed Miniconda
- Located at:
    
    ```
    E:\miniconda3
    
    ```
    
- Contains:
    - Conda itself
    - A minimal Python

You generally:

- Use `base` to **create and manage other environments**
- Avoid installing lots of project packages into `base`

## Typical daily workflow (simple)

```bash
conda activate base        # once per session
conda create -n research python=3.11
conda activate research
python

```

---

## Key takeaway

- `conda activate base` = **turn Conda on**
- `(base)` = Conda is active
- No `(base)` = Conda is idle

## Optional: put environments somewhere else

If you want to **always store environments on E:** separately from Miniconda root:

```bash
conda config --add envs_dirs E:\conda_envs

```

- Now all future environments will go to:

```
E:\conda_envs\<env name>

```

- Clean separation → easy to manage → C: remains free

**Conda**: manages **Python installations and environments**

- Each environment = its own isolated Python + packages
- Keeps your projects independent

**VS Code**: an **editor/IDE**, not Python itself

- Needs to know which Python to use
- Needs extra functionality (linting, debugging, notebooks)

**Together**:

- VS Code reads your Conda environment’s Python
- Runs code / notebooks inside that environment
- Allows you to switch between multiple Conda environments per projecBig picture

Think of it as **three layers working together**:

1. **Conda** → Python runtime + isolated environments
2. **Python packages** → NumPy, Pandas, Jupyter, etc. installed in Conda env
3. **VS Code + Python extension + Pylance** → IDE features, environment selection, notebooks, debugging

All three together give you a **professional, clean workflow**.

### Remove old interpreter cache

1. Open Command Palette: `Ctrl+Shift+P`
2. Type: `Python: Clear Workspace Interpreter Setting`
3. Restart VS Code

> This forces VS Code to only use valid interpreters.
> 

## Hide base from selection (if you want)

- Open Command Palette → `Python: Select Interpreter` → `Enter interpreter path` → choose **ml**
- In workspace `.vscode/settings.json`, it will store:

```json
{
    "python.defaultInterpreterPath": "E:\\miniconda3\\envs\\ml\\python.exe"
}

```

- This ensures VS Code always uses `ml` for this project
- Base will still exist, but you **won’t have to select it manually**

## Optional cleanup (safe)

To save space later:

```powershell
conda clean --all

```

(Removes cached packages only.)

## Your workflow (simple & clean)

```
VS Code
 ├─ Conda env (ml)
 ├─ research-notebook/
 │   ├─ README.md
 │   ├─ notebooks/
 │   └─ notes/
 └─ Git (push to GitHub)

```

No extra tools. No clutter.

1. Move all files/folders to the parent folder:

```powershell
Move-Item .git .. -Force
Move-Item * .. -Force
Move-Item .* .. -Force 2>$null

```

- This moves the hidden `.git` folder to the **parent folder**
- Second line moves all normal files
- `Force` ensures it overwrites if needed
- `2>$null` suppresses harmless errors about `.` and `..`
- Close any programs using the subfolder

## Step 3 — Go back to root and delete empty subfolder

```powershell
cd ..
rmdir research-notebook

```

---

## Step 4 — Verify everything

```powershell
ls -Force
git status

```

✅ You should now see:

- `.git` folder
- `README.md`
- any other files/folders
- **no nested `research-notebook` folder**
- Git should work normally