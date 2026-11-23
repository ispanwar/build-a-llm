# Build a LLM From Scratch

A hands-on project where I build a large language model from the ground up in PyTorch. It follows the book *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

## Progress

| Chapter | Topic | File |
|---------|-------|------|
| Intro | PyTorch basics | `notebooks/torch_intro.py` |
| 2 | Working with text data | `notebooks/chapter_2_working_with_text_data.py` |
| 3 | Coding self-attention | `notebooks/chapter_3_coding_self_attention.py` |

Sample text used in the exercises: `notebooks/the-verdict.txt`.

## Project Structure

```
build-a-llm/
├── notebooks/        # Chapter-by-chapter code
├── requirements.txt  # Python dependencies
└── README.md
```

## Setup

Requires Python 3.11.

```powershell
# Create and activate a virtual environment
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

If PowerShell blocks activation, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Usage

Launch JupyterLab and open the files in `notebooks/`:

```powershell
jupyter lab
```

Or run a script directly:

```powershell
python notebooks/chapter_2_working_with_text_data.py
```

## Reference

- Book: *Build a Large Language Model (From Scratch)*, Sebastian Raschka
- Official code: https://github.com/rasbt/LLMs-from-scratch
