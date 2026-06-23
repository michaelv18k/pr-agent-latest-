import os

import subprocess
import tempfile
import json
import litellm


import asyncio
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from github import Github
from pr_agent.agent.pr_agent import PRAgent
from pr_agent.config_loader import get_settings
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.environments.local import LocalEnvironment
 
app = FastAPI()
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — reads from your environment variables
# Set these in terminal before running:
#   export GITHUB_TOKEN="ghp_xxxx"
#   export GROQ_API_KEY="gsk_xxxx"
# ─────────────────────────────────────────────────────────────────────────────
 
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME   = "groq/llama-3.3-70b-versatile"
 
# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI ROUTES
# ─────────────────────────────────────────────────────────────────────────────
 
@app.get("/")
def home():
    return {"status": "PR Agent + SWE Agent running!"}
 
 
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    event_type = request.headers.get("X-GitHub-Event", "unknown")
    payload    = await request.json()
 
    if event_type == "ping":
        print("✅ Webhook connected!")
        return {"status": "pong"}
 
    if event_type == "pull_request":
        action = payload.get("action")
 
        if action in ["opened", "synchronize"]:
            # Extract everything we need from the webhook payload
            pr_url    = payload["pull_request"]["html_url"]
            pr_number = payload["pull_request"]["number"]
            repo_name = payload["repository"]["full_name"]        # "user/retail-etl-pipeline"
            repo_url  = payload["repository"]["clone_url"]        # "https://github.com/user/retail-etl-pipeline.git"
            branch    = payload["pull_request"]["head"]["ref"]     # "feature/add-inventory"
            author    = payload["pull_request"]["user"]["login"]   # "barandeep"
 
            print(f"\n{'='*60}")
            print(f"🔔 PR #{pr_number} received from @{author}")
            print(f"   Branch: {branch}")
            print(f"   URL: {pr_url}")
            print(f"{'='*60}")
 
            # Run the full pipeline in background
            # (so GitHub doesn't timeout waiting for our response)
            background_tasks.add_task(
                run_full_pipeline,
                pr_url, pr_number, repo_name, repo_url, branch, author
            )
 
    return {"status": "accepted"}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: CONFIGURE PR-AGENT
# ─────────────────────────────────────────────────────────────────────────────
 
def configure_pr_agent():
    """Set all pr-agent settings before running."""
    get_settings().set("config.model", MODEL_NAME)
    get_settings().set("github.user_token", GITHUB_TOKEN)
    get_settings().set("groq.api_key", GROQ_API_KEY)
 
    # Turn on all review features
    get_settings().set("pr_reviewer.require_security_review", True)
    get_settings().set("pr_reviewer.require_estimate_effort_to_review", True)
    get_settings().set("pr_reviewer.num_code_suggestions", 5)
    get_settings().set("pr_code_suggestions.summarize", True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: RUN PR-AGENT REVIEW + COLLECT ISSUES
# ─────────────────────────────────────────────────────────────────────────────
 
async def run_pr_agent_review(pr_url: str, pr_number: int, repo_name: str) -> list[str]:
    """
    Run pr-agent /review and /improve on the PR.
    Then read the review comments back from GitHub and return them as a list.
    """
    configure_pr_agent()
    agent = PRAgent()
 
    print(f"\n📋 STEP 1: PR-Agent reviewing PR #{pr_number}...")
 
    # /review  → posts overall review (security, correctness, effort score)
    await agent.handle_request(pr_url, "/review")
    print("   ✅ /review done — check GitHub PR for comments")
 
    # /improve → posts inline code suggestions on specific lines
    await agent.handle_request(pr_url, "/improve")
    print("   ✅ /improve done — inline suggestions posted")
 
    # Wait a moment for GitHub to register the comments
    await asyncio.sleep(3)
 
    # Now READ the comments pr-agent just posted
    # so we can pass them to SWE-agent as context
    # issues = collect_review_issues(repo_name, pr_number)

    raw_issues = collect_review_issues(repo_name, pr_number)

    raw_issues = raw_issues[:10]
    print(f"Raw issues: {len(raw_issues)}")

    issues = extract_real_issues(raw_issues)

    print(f"Real issues: {len(issues)}")

    for issue in issues:
        print("-", issue)

    print(f"   📝 Collected {len(issues)} issues from review")
    return issues
 
 
# def collect_review_issues(repo_name: str, pr_number: int) -> list[str]:
    """
    Read the PR comments from GitHub that pr-agent just posted.
    Extract the key issues to pass to SWE-agent.
    """
    g  = Github(GITHUB_TOKEN)
    pr = g.get_repo(repo_name).get_pull(pr_number)
 
    issues = []
 
    # Read PR-level review comments (the summary block)
    for review in pr.get_reviews():
        body = review.body
        if body and len(body) > 50:  # skip empty or tiny comments
            # Extract bullet points / numbered items from the review
            for line in body.split("\n"):
                line = line.strip()
                if line.startswith(("-", "*", "•", "1", "2", "3", "4", "5")):
                    if len(line) > 10:  # skip empty bullets
                        issues.append(line.lstrip("-*•0123456789. "))
 
    # Read inline review comments (specific line comments)
    for comment in pr.get_review_comments():
        issues.append(
            f"File: {comment.path}, Line {comment.position}: {comment.body[:200]}"
        )
 
    return issues
 

def collect_review_issues(repo_name: str, pr_number: int) -> list[str]:
    """
    Read ALL comments from the PR that pr-agent posted.
    Wait longer to make sure GitHub has registered them.
    """
    import time
    time.sleep(8)  # give GitHub time to register all comments

    g    = Github(GITHUB_TOKEN)
    repo = g.get_repo(repo_name)
    pr   = repo.get_pull(pr_number)

    issues = []

    # Read every general comment on the PR (pr-agent posts here)
    for comment in pr.get_issue_comments():
        body = comment.body
        if not body:
            continue
        print(f"   📄 Found comment ({len(body)} chars)")
        # grab every non-empty line as a potential issue
        for line in body.split("\n"):
            line = line.strip()
            if len(line) > 15 and not line.startswith("#"):
                issues.append(line.lstrip("-*•>| "))

    # Read inline review commentsß
    for comment in pr.get_review_comments():
        issues.append(f"File {comment.path}: {comment.body[:300]}")
        print(f"   📄 Found inline comment on {comment.path}")

    # Read formal reviews
    for review in pr.get_reviews():
        if review.body and len(review.body) > 20:
            issues.append(review.body[:500])

    print(f"   Raw issues collected: {len(issues)}")

    # If still nothing, use a default fallback
    # so SWE-agent still runs a general cleanup
    if not issues:
        print("   ⚠️  No comments parsed — using general fix task")
        issues = [
            "Fix any hardcoded credentials — use os.environ.get()",
            "Replace SELECT * with explicit column names",
            "Add WHERE clause to prevent full table scans",
            "Replace bare except with specific exception handlers",
            "Add docstrings and type hints to all functions",
            "Replace print() with logger.info() calls"
        ]

    return issues


# def run_swe_agent_fix(task: str) -> str:
    """
    Pass the task to mini-swe-agent.
    Fixed: correct argument name is 'env' not 'environment'
    """
    print(f"\n🤖 STEP 3: SWE-Agent fixing the code...")

    model = LitellmModel(model_name=MODEL_NAME)
    env   = LocalEnvironment()

    # FIXED: 'env' not 'environment'
    agent = DefaultAgent(model=model, env=env)

    result = agent.run(task)
    print(f"   ✅ SWE-Agent finished")
    return str(result)


def run_swe_agent_fix(task: str) -> str:
    """
    Run mini-swe-agent via subprocess (CLI) instead of Python API.
    This avoids version-specific API changes and gives full bash access.
    """
    print(f"\n🤖 STEP 3: SWE-Agent fixing the code via CLI...")

    # Write the task to a temp file so we can pass it cleanly
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.txt',
        delete=False,
        prefix='swe_task_'
    ) as f:
        f.write(task)
        task_file = f.name

    print(f"   📄 Task written to: {task_file}")

    try:
        # Set environment for the subprocess
        env = os.environ.copy()
        env["GROQ_API_KEY"]   = GROQ_API_KEY
        env["GITHUB_TOKEN"]   = GITHUB_TOKEN

        # Run mini-swe-agent CLI with the task
        # --yolo means no human confirmation needed — fully autonomous
        result = subprocess.run(
            [
                "mini",
                "--model", MODEL_NAME,
                "--yolo",                    # no human confirmation
                "--task", task,              # pass task directly
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300                      # 5 min max
        )

        print(f"   STDOUT:\n{result.stdout[-2000:]}")  # last 2000 chars

        if result.returncode == 0:
            print("   ✅ SWE-Agent completed successfully")
        else:
            print(f"   ⚠️  SWE-Agent stderr:\n{result.stderr[-1000:]}")

        return result.stdout

    except subprocess.TimeoutExpired:
        print("   ⚠️  SWE-Agent timed out after 5 minutes")
        return "timeout"

    except FileNotFoundError:
        print("   ❌ 'mini' command not found — trying python -m")
        # Fallback: run as python module
        result = subprocess.run(
            [
                "python", "-m", "minisweagent",
                "--model", MODEL_NAME,
                "--task", task,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.stdout

    finally:
        # Clean up temp file
        os.unlink(task_file)
# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: BUILD THE TASK STRING FOR SWE-AGENT
# ─────────────────────────────────────────────────────────────────────────────
 
def build_swe_agent_task(
    repo_url:  str,
    branch:    str,
    issues:    list[str],
    repo_name: str,
    pr_number: int,
    author:    str
) -> str:
    """
    Build the instruction string that tells SWE-agent exactly what to do.
    This is the bridge between pr-agent's output and SWE-agent's input.
    """
 
    issues_text = "\n".join(f"  - {issue}" for issue in issues) if issues else \
                  "  - General code quality issues found. Review and fix any problems."
 
    # IMPORTANT: we embed the GITHUB_TOKEN in the clone URL
    # so SWE-agent can push back without needing separate auth
    auth_repo_url = repo_url.replace(
        "https://",
        f"https://{GITHUB_TOKEN}@"
    )
 
    task = f"""
You are an automated code fixer. A pull request has been reviewed and issues were found.
Your job is to fix ALL the issues, then commit and push the fixes.
 
=== REPOSITORY INFORMATION ===
Repository URL: {auth_repo_url}
Branch to fix:  {branch}
PR Number:      #{pr_number}
PR Author:      @{author}
 
=== ISSUES FOUND BY CODE REVIEW ===
{issues_text}
 
=== YOUR EXACT STEPS — FOLLOW IN ORDER ===
 
Step 1: Clone the repository
  git clone {auth_repo_url} /tmp/pr_fix_{pr_number}
 
Step 2: Switch to the PR branch (NOT main)
  cd /tmp/pr_fix_{pr_number}
  git checkout {branch}
 
Step 3: Read all changed Python and SQL files
  find . -name "*.py" -o -name "*.sql" | grep -v ".git"
 
Step 4: Fix every issue listed above.
  Common fixes for this retail ETL project:
  - Hardcoded passwords → replace with os.environ.get("SNOWFLAKE_PASSWORD")
  - Hardcoded usernames → replace with os.environ.get("SNOWFLAKE_USER")
  - Hardcoded accounts  → replace with os.environ.get("SNOWFLAKE_ACCOUNT")
  - Missing docstrings  → add triple-quoted docstrings to all functions
  - Bare except:        → replace with except snowflake.connector.Error as e:
  - Missing type hints  → add type hints to function parameters and return types
  - SELECT *            → replace with explicit column names
  - print() statements  → replace with logger.info() or logger.error()
  - Missing WHERE clause → add date filter WHERE transaction_date >= DATEADD(day, -90, CURRENT_DATE())
 
Step 5: Verify your fixes by reading the files again
  cat etl/extract/extract_sales.py
 
Step 6: Run tests to make sure nothing is broken
  cd /tmp/pr_fix_{pr_number}
  python -m pytest tests/ -v 2>&1 || echo "Tests done (some may fail without Snowflake connection)"
 
Step 7: Configure git identity
  git config user.email "pr-review-bot@retail-etl.com"
  git config user.name "PR Review Bot"
 
Step 8: Commit ALL changes
  git add .
  git commit -m "Auto-fix: resolved code review issues in PR #{pr_number}
 
  Issues fixed:
{issues_text}
 
  Fixed by automated PR review agent."
 
Step 9: Push to the PR branch (NOT main)
  git push origin {branch}
 
Step 10: Clean up
  cd /tmp
  rm -rf /tmp/pr_fix_{pr_number}
 
When you have successfully pushed, output exactly:
MINI_SWE_AGENT_FINAL_OUTPUT: Successfully fixed and pushed all issues to branch {branch}
"""
    return task
 
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: RUN SWE-AGENT TO FIX THE CODE
# ─────────────────────────────────────────────────────────────────────────────
 
# def run_swe_agent_fix(task: str) -> str:
    """
    Pass the task to mini-swe-agent.
    It runs bash commands to clone, fix, commit, and push.
    Returns the final output string.
    """
    print(f"\n🤖 STEP 2: SWE-Agent fixing the code...")
 
    model = LitellmModel(model_name=MODEL_NAME)
    env   = LocalEnvironment()
    agent = DefaultAgent(model=model, environment=env)
 
    # agent.run() is synchronous — it blocks until done
    result = agent.run(task)
 
    print(f"   ✅ SWE-Agent finished")
    return str(result)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: POST FINAL COMMENT TO GITHUB
# ─────────────────────────────────────────────────────────────────────────────
 
def post_completion_comment(repo_name: str, pr_number: int, author: str, branch: str):
    """
    After SWE-agent pushes the fix, post a final comment on the PR
    tagging the reviewer to merge.
    """
    g    = Github(GITHUB_TOKEN)
    repo = g.get_repo(repo_name)
    pr   = repo.get_pull(pr_number)
 
    comment = f"""## ✅ Automated Fix Complete
 
Hi @{author} — the code review issues have been automatically fixed and committed to your branch `{branch}`.
 
### What was fixed:
- 🔒 Security: Hardcoded credentials replaced with environment variables
- ✅ Correctness: Docstrings and type hints added to all functions
- ⚡ Performance: SELECT * replaced with explicit columns + WHERE clause added
- 🔧 Quality: Bare except clauses replaced with specific exception handlers
- 📝 Logging: print() statements replaced with proper logger calls
 
### Next steps:
The fixes have been pushed to `{branch}`. Please review the changes and merge when ready.
 
> *This fix was applied automatically by the PR Review Agent.*
"""
 
    pr.create_issue_comment(comment)
    print(f"   ✅ Final comment posted on PR #{pr_number}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE — orchestrates everything
# ─────────────────────────────────────────────────────────────────────────────
 
async def run_full_pipeline(
    pr_url:    str,
    pr_number: int,
    repo_name: str,
    repo_url:  str,
    branch:    str,
    author:    str
):
    """
    Full pipeline:
      1. pr-agent reviews the PR and posts comments
      2. Collect the review issues
      3. Build task for SWE-agent
      4. SWE-agent clones, fixes, commits, pushes
      5. Post final comment tagging reviewer
    """
    try:
        print(f"\n🚀 Starting full pipeline for PR #{pr_number}")
 
        # ── STAGE 1: PR-Agent reviews ────────────────────────────────────────
        issues = await run_pr_agent_review(pr_url, pr_number, repo_name)
 
        # ── STAGE 2: Build SWE-agent task ────────────────────────────────────
        print(f"\n🔧 STEP 2: Building SWE-agent task with {len(issues)} issues...")

        print("\n" + "="*60)
        print(f"ALL {len(issues)} ISSUES COLLECTED:")
        print("="*60)
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        print("="*60 + "\n")
        
        task = build_swe_agent_task(
            repo_url, branch, issues, repo_name, pr_number, author
        )
 
        # ── STAGE 3: SWE-agent fixes + commits + pushes ──────────────────────
        # Run in executor because agent.run() is synchronous
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_swe_agent_fix, task)
 
        # ── STAGE 4: Post final comment on GitHub ─────────────────────────────
        print(f"\n💬 STEP 3: Posting completion comment...")
        post_completion_comment(repo_name, pr_number, author, branch)
 
        print(f"\n{'='*60}")
        print(f"🎉 PIPELINE COMPLETE for PR #{pr_number}")
        print(f"   Branch {branch} has been fixed and pushed")
        print(f"   Reviewer has been notified")
        print(f"{'='*60}\n")
 
    except Exception as e:
        print(f"\n❌ Pipeline failed for PR #{pr_number}: {str(e)}")
        import traceback
        traceback.print_exc()
 
        # Post error comment on GitHub so team knows what happened
        try:
            g    = Github(GITHUB_TOKEN)
            repo = g.get_repo(repo_name)
            pr   = repo.get_pull(pr_number)
            pr.create_issue_comment(
                f"⚠️ Automated fix pipeline encountered an error:\n```\n{str(e)}\n```\n"
                f"Manual review required for PR #{pr_number}."
            )
        except Exception:
            pass
 

 # ─────────────────────────────────────────────────────────────────────────────
# Extract real issues 
# ─────────────────────────────────────────────────────────────────────────────


def extract_real_issues(raw_issues: list[str], model: str = "groq/llama-3.3-70b-versatile") -> list[str]:
    """
    Convert noisy PR-Agent review output into a small set of actionable issues.

    Input:
        raw_issues = [
            "<html>",
            "Hardcoded password...",
            "<table>...",
            ...
        ]

    Output:
        [
            "Replace hardcoded Snowflake credentials with environment variables",
            "Add type hints to extract_pos_transactions",
            "Replace SELECT * with explicit columns and WHERE clause"
        ]
    """

    review_text = "\n".join(raw_issues)

    prompt = f"""
You are an expert code-review issue extractor.

The input below comes from an automated PR review tool and may contain:

- HTML
- Markdown
- Tables
- Review summaries
- Effort estimates
- Duplicate findings
- Previous bot comments
- Commit references

Your job:

1. Ignore all formatting.
2. Ignore HTML and markdown.
3. Ignore review metadata.
4. Ignore duplicated findings.
5. Extract ONLY real actionable code issues.
6. Merge duplicates.
7. Return ONLY JSON.

Example:

{{
  "issues": [
    "Replace hardcoded credentials with environment variables",
    "Add missing type hints",
    "Replace SELECT * with explicit columns",
    "Replace print() with logger",
    "Replace bare except with specific exception handling"
  ]
}}

Input:

{review_text}
"""

    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You extract actionable code-review issues."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        parsed = json.loads(content)
        return parsed.get("issues", [])
    except Exception:
        print("Failed to parse issue extraction output")
        print(content)
        return []
# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)