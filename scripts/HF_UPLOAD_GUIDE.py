"""
Hugging Face Model Upload Guide
================================

IMPORTANT: This script only UPLOADS a copy of your model.
It does NOT modify your local project in any way!

Steps to Upload:
================

1. Create Hugging Face Account (✓ You already did this!)
   - Go to https://huggingface.co

2. Create an Access Token:
   a. Go to https://huggingface.co/settings/tokens
   b. Click "New token"
   c. Name: "RecipeAI Upload"
   d. Type: Select "Write" access
   e. Click "Generate"
   f. COPY THE TOKEN (you'll need it below)

3. Install Hugging Face CLI:
   ```bash
   cd /home/bhavesh/MajorB/RecipeAI
   source venv/bin/activate
   pip install huggingface_hub
   ```

4. Login to Hugging Face:
   ```bash
   huggingface-cli login
   ```
   (Paste your token when prompted)

5. Run the upload script:
   ```bash
   python scripts/upload_to_hf.py
   ```

What Gets Uploaded:
===================
- Your trained model (recipe_agent_standard_ppo.zip)
- Model card with description
- Usage instructions
- Training stats

What DOESN'T Change:
====================
✓ Your local model stays in models/saved/
✓ All your code stays the same
✓ No files are deleted or modified
✓ Your project keeps working exactly as before

The upload creates a COPY on Hugging Face's servers.

After Upload:
=============
You'll get a link like:
https://huggingface.co/YOUR_USERNAME/recipe-generation-rl

You can share this with your friend, and they can download with:
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/recipe-generation-rl",
    filename="recipe_agent_standard_ppo.zip"
)
```

Benefits:
=========
✓ Easy sharing
✓ Version control (can upload updates later)
✓ Backup of your trained model
✓ Citation and documentation
✓ Community can use your model

Questions?
==========
Just ask! This is completely safe and non-destructive.
"""

print(__doc__)
