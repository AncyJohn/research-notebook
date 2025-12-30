# Phase 1 – Installation Journal

# 

**Date: 30/12/2025**

**Phase:** Data Science Career Restart – Phase 1 (Foundation)

---

## 1. Purpose of This Journal

This document records all steps taken during the environment setup for my data science and deep learning restart. It serves as:

- A personal reference
- A reproducibility log
- A research-style habit (documenting process, not just results)

---

## 2. System Information

- **Operating System:**
- **Machine Type:** (Laptop/Desktop)
- **RAM:**
- **CPU:**
- **GPU:** (if any)

---

## 3. GitHub Setup

### 3.1 GitHub Account

- Existing GitHub account reused
- Repository created: `research-notebook`
- Repository visibility: Public

### 3.2 Repository Purpose

The `research-notebook` repository will be used as:

- A research diary
- A record of experiments
- A place for learning notes, Kaggle work, and paper reproductions

---

## 4. Miniconda Installation

### 4.1 Reason for Choosing Miniconda

- Lightweight
- Environment isolation
- Widely used in ML research

### 4.2 Installation Steps

1. Removed the left over old anaconda files from the D drive
2. Downloaded Miniconda (Python 3.10) on drive E
3. Installed using default settings (Do not add to path, allow cache cleaning & default python)
4. Verified installation via terminal (Anaconda prompt starts in C since windows terminal default to your user home directory.)

### 4.3 Environment Creation

```bash
conda info
conda --version
conda create -p E:\conda_envs\ml python=3.10
conda activate E:\conda_envs\ml

```

Using `-p` ensures the **full path is set** and avoids issues with spaces
**Result:** Environment activated successfully (`(`E:\conda_envs\`ml)` visible in terminal)

---

## 5. VS Code Installation

1. Go to the official website: [VS Code Download](https://code.visualstudio.com/download)
2. Choose **Windows → User Installer (64-bit)** — this installs for your user only, **no admin needed**.
3. Run the installer, select:
- **Add “Open with Code” to context menu** ✅
    - Open with Code action to Windows Explorer file context menu
    - Open with Code action to Windows Explorer directory context menu
- **Add to PATH** ✅
- Register Code as an editor for supported file types
- **Do not install other features you don’t need**

### 5.1 Reason for Choosing VS Code

- Strong Python & Jupyter support
- GitHub integration
- Industry standard

### 5.2 Extensions Installed

- Python
- Jupyter
- Pylance

### To overcome PowerShell security restriction

## Step 1 — Open PowerShell **as yourself** (NOT admin)

- Close VS Code
- Open **PowerShell** normally (Start → PowerShell)

---

## Step 2 — Set execution policy for your user

Run **exactly this**:

## To overcome PowerShell security restriction

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

```

When prompted:

```
Do you want to change the execution policy?

```

Type:

```
Y

```

Press **Enter**

## Test in the terminal inside VS Code

1. Open **Terminal** in VS Code: `Ctrl + ```
2. Run:

```bash
where python
python --version
conda info --envs

```

Expected output:

```
E:\conda_envs\ml\python.exe
Python 3.10.x
# conda environments:
#
base                  E:\miniconda3
ml                    E:\conda_envs\ml *

```

- indicates the **currently active environment**
- `where python` should show the path to your `ml` environment first

---

## 4️⃣ Test a simple Python import

In VS Code terminal or a new script:

```python
import sys
print(sys.executable)

```

Expected output:

```
E:\conda_envs\ml\python.exe

```

This **proves VS Code is actually using your Conda environment**.

## Connect VS Code to your Conda environment

1. Open **Command Palette**: `Ctrl+Shift+P`
2. Search: `Python: Select Interpreter`
3. Choose your **ML environment** Python path, e.g.:

```
E:\conda_envs\ml\python.exe

```

Now, **all Python code and notebooks** in VS Code will use your clean Conda environment.

## Optional: Create a workspace for your projects

- Go to **File → Open Folder → E:\projects** (or your project folder)
- VS Code will remember the environment per workspace

This keeps **everything tidy and reproducible**.

---

## 6. Git & Repository Setup

### 6.1 Tools Used

- Git

## Step-by-step on GitHub (10 minutes max)

### 1️⃣ Go to GitHub

Open: [https://github.com](https://github.com/)

Make sure you are **logged in**.

---

### 2️⃣ Create a new repository

Click **➕ (top-right)** → **New repository**

---

### 3️⃣ Fill the repository form

### 🔹 Repository name

```
research-notebook

```

⚠️ Spelling matters — use **exactly this**

---

### 🔹 Description (optional but recommended)

You can paste:

```
Personal research notebook for experiments, notes, and learning logs.

```

(or leave it blank)

---

### 🔹 Visibility

Select:

- ✅ **Public**

---

### 🔹 Initialize repository (important)

Check:

- ☑️ **Add a README file**

Do **NOT** check:

- ⛔ Add .gitignore (leave it for later)
- ⛔ Choose a license (**leave as “None”**)

> GitHub defaults to no license if you don’t pick one — you don’t need to do anything special.
> 

---

### 4️⃣ Create repository

Click:

👉 **Create repository**

That’s it 🎉

## Step 1 — Download Git for Windows

1. Open your browser
2. Go to: [**https://git-scm.com**](https://git-scm.com/)
3. Click **Download for Windows**
4. Run the installer (`.exe`)

---

## Step 2 — Installer options (IMPORTANT)

I’ll list **only what to select / not select**.

Everything else → **leave default**.

---

### 🔹 Select Components

✅ Check:

- **Git Bash Here**
- **Git GUI Here**

❌ Optional (leave unchecked):

- Git LFS (you can add later)
- bundled OpenSSH

Click **Next**

---

### 🔹 Choosing the default editor

Select:

```
Use Visual Studio Code as Git’s default editor

```

✔️ This is important for commit messages.

---

### 🔹 Adjusting PATH environment

Select:

```
Git from the command line and also from 3rd-party software

```

✔️ This lets VS Code + PowerShell see Git.

---

### 🔹 Choosing HTTPS transport backend

Leave default:

```
Use the OpenSSL library

```

---

### 🔹 Configuring line ending conversions

Select:

```
Checkout Windows-style, commit Unix-style line endings

```

---

### 🔹 Configuring terminal emulator

Select:

```
Use Windows' default console window

```

---

### 🔹 Default behavior of `git pull`

Leave default:

```
Fast-forward or merge

```

---

### 🔹 Credential helper

Leave default:

```
Git Credential Manager

```

✔️ This avoids typing passwords.

---

### 🔹 Extra options

Leave everything default.

---

### 🔹 Experimental options

❌ Leave unchecked.

---

## Step 3 — Finish & restart terminals

After install:

- Close **VS Code**
- Reopen **VS Code**
- Open **Terminal → New Terminal**

Run:

```powershell
git --version

```

Expected:

```
git version 2.xx.x

```

---

## Step 4 — One-time Git identity setup

In VS Code terminal:

```powershell
git config --global user.name "Ancy Antony"
git config --global user.email "your-github-email@example.com"

```

⚠️ Use the **same email as your GitHub account**.

### 6.2 Steps

1. Cloned `research-notebook` repository locally
2. Opened repository folder in VS Code
3. Created initial `README.md`
4. Committed and pushed changes

## Clone your repo locally

1. Open VS Code → Terminal → New Terminal
2. Navigate to a folder where you want the repo:

```powershell
mkdir E:\projects
cd E:\projects

```

*(create `projects` if you like: `mkdir E:\projects`)*

1. Clone your GitHub repo:

```powershell
git clone https://github.com/<your-username>/research-notebook.git

```

Replace `<your-username>` with your GitHub username.

You’ll see a folder:

```
E:\projects\research-notebook\

```

---

## 2️⃣ Open the repo in VS Code

- File → Open Folder → select `E:\projects\research-notebook`
- Terminal will open at the repo root

---

## 3️⃣ Connect your Conda environment

1. Press **Ctrl + Shift + P** → `Python: Select Interpreter`
2. Choose your environment path:

```
E:\conda_envs\ml\python.exe

```

1. Open a terminal → check:

```powershell
python --version

```

Expected: `Python 3.10.x`

---

## 7. Initial README Content

```markdown
# Research Notebook

This repository documents my journey restarting in data science and deep learning.

Focus areas:
- Machine Learning fundamentals
- Deep Learning experiments
- Kaggle competitions
- Research notes and reproductions

This is a learning-first, research-style notebook.

```

---

## Step 1 — Open VS Code terminal in the repo

Make sure your terminal is **inside your repository**:

```powershell
pwd

```

Expected output:

```
Path
----
E:\projects\research-notebook

```

---

## Step 2 — Check Git status

## Step 1 — Save your changes

- In VS Code, press **Ctrl + S** (or File → Save) while the README.md file is open
- Make sure it’s the **README.md in the root folder**:

```
E:\projects\research-notebook\README.md

```

```powershell
git status

```

You should see:

```
Untracked files:
  README.md

```

> This tells Git that README.md exists locally but hasn’t been committed yet.
> 

---

## Step 3 — Stage the README file

```powershell
git add README.md

```

- `git add` tells Git: *“track this file for the next commit”*

---

## Step 4 — Commit the changes

```powershell
git commit -m "Add initial README"

```

- `m` adds a message describing the commit
- Now Git has **recorded the change locally**

---

## Step 5 — Push to GitHub

```powershell
git push origin main

```

- `origin` → the GitHub repository you cloned from
- `main` → default branch name (could be `master` if your repo uses that)

> You may be prompted for GitHub credentials (username/email or a personal access token).
> 

---

## Step 6 — Verify

- Go to GitHub in your browser
- Open your **`research-notebook`** repo
- You should now see **README.md** updated

---

## ✅ Optional: One-line shortcut

```powershell
git add README.md && git commit -m "Add initial README" && git push origin main

```

- This stages, commits, and pushes in **one command**

Finally authorize your push command on git hub window.

## 8. Reflections (Important)

- Restarting after a long gap felt intimidating, but breaking the process into small steps helped
- Focus is intentionally kept on research and deep learning, not visualization-heavy work
- The goal is consistency and curiosity, not immediate results

---

## 9. Next Steps

- Install ML libraries
- Set up Jupyter Notebook workflow
- Create first research notebook

---

**Status:** Phase 1 – Day 1 completed successfully