import subprocess
import httpx
import sys

username = "ANJULS09"
token = "@Pagesecurity1@"
repo_name = "customer-voice-intelligence"

# URL encode password/token for the remote URL (especially the '@' symbol)
encoded_token = token.replace("@", "%40")

# 1. Attempt to create the repository on GitHub
print("Attempting to create the remote repository on GitHub via API...")
url = "https://api.github.com/user/repos"
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}
payload = {
    "name": repo_name,
    "private": False,
    "description": "Enterprise Grounded AI Insights Engine for Havells reviews analysis."
}

try:
    r = httpx.post(url, headers=headers, json=payload, timeout=15.0)
    if r.status_code == 201:
        print(f"-> Successfully created repository '{repo_name}' on GitHub!")
    elif r.status_code == 422:
        print(f"-> Note: Repository '{repo_name}' may already exist or there was a validation check.")
    elif r.status_code == 401:
        print("-> API Error 401: Unauthorized. Note: If this is your account password, GitHub requires a Personal Access Token (PAT) for API and Git actions.")
    else:
        print(f"-> API Check: Status {r.status_code}: {r.text}")
except Exception as e:
    print(f"-> GitHub API check failed: {e}")

# 2. Local git execution
print("\nInitializing local git repo and committing files...")
git_commands = [
    ("git init", "Initializing git"),
    ("git config --local user.name \"ANJULS09\"", "Setting git user name"),
    ("git config --local user.email \"anjuls09@users.noreply.github.com\"", "Setting git email"),
    ("git add .", "Staging project files"),
    ("git commit -m \"Initial commit: Customer Voice Intelligence Agent (CVIA) enterprise system\"", "Committing staged files"),
    ("git branch -M main", "Rename branch to main"),
    # Remove existing remote if it exists
    ("git remote remove origin", "Cleaning old remote"),
    (f"git remote add origin https://{username}:{encoded_token}@github.com/{username}/{repo_name}.git", "Adding remote origin"),
    ("git push -u origin main --force", "Pushing to remote repository")
]

for cmd, desc in git_commands:
    print(f"Running: {desc}...")
    # Hide token in print statements
    visible_cmd = cmd
    if token in cmd:
        visible_cmd = cmd.replace(token, "[REDACTED_SECRET]").replace(encoded_token, "[REDACTED_SECRET]")
    
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        # Ignore remote remove failures since remote may not exist
        if "remote remove" in cmd:
            continue
        print(f"-> Error: {res.stderr.strip()}")
        if "push" in cmd:
            print("\nIf the push failed due to authentication, please verify that your password is a GitHub Personal Access Token (PAT) with 'repo' scope permissions.")
            sys.exit(1)
    else:
        print("-> Success.")

print("\nGit push process completed!")
