import asyncio
import re
from typing import Dict, Any, Union, List
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from pr_agent.agent.pr_agent import PRAgent
from pr_agent.config_loader import get_settings

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI(title="PR-Agent & Aider-Agent Integration Server")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"[ERROR] Validation error: {exc.errors()}")
    try:
        body = await request.json()
        print(f"[ERROR] Request body: {body}")
    except Exception:
        body_bytes = await request.body()
        print(f"[ERROR] Request body (raw): {body_bytes}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body)}
    )

# In-memory coordination database
# Key: pr_number (str) -> Value: {"pr_url": str, "event": asyncio.Event(), "findings": Any}
pending_reviews: Dict[str, Dict[str, Any]] = {}

class FindingItem(BaseModel):
    suggestion_summary: str = None
    relevant_file: str = None
    relevant_lines_start: Union[int, str, None] = None
    relevant_lines_end: Union[int, str, None] = None
    suggestion_score: Union[int, str, None] = None
    why: str = None
    severity: str = None

class FindingsPayload(BaseModel):
    pr_id: Union[int, str]
    source: str = None
    findings: Union[List[FindingItem], List[Dict[str, Any]], str]

class IncomingSuggestionItem(BaseModel):
    file_path: str
    line_number: Union[int, str, None] = None
    severity: str = None
    category: str = None
    description: str = None
    suggestion: str = None
    confidence: Union[float, int, None] = None

class IncomingFindingsPayload(BaseModel):
    pr_id: Union[int, str]
    source: str = None
    my_suggestions: List[IncomingSuggestionItem]

def extract_pr_number(pr_url: str) -> str:
    """Extracts the PR number/id from a GitHub or Azure DevOps PR URL (e.g. .../pull/6 or .../pullrequest/6 -> '6')."""
    match = re.search(r'/pull/(\d+)', pr_url)
    if match:
        return match.group(1)
    match = re.search(r'/pullrequest/(\d+)', pr_url, re.IGNORECASE)
    if match:
        return match.group(1)
    return pr_url

def extract_pr_url(body: dict) -> str:
    """Helper to extract PR URL from a generic, GitHub, or Azure DevOps webhook body."""
    if "pr_url" in body:
        return body["pr_url"]
    if "pull_request" in body:
        pull_req = body["pull_request"]
        return pull_req.get("html_url") or pull_req.get("url")
    
    # Azure DevOps webhook integration
    if "eventType" in body and "resource" in body:
        from urllib.parse import unquote
        resource = body["resource"]
        event_type = body.get("eventType")
        
        # Check for pull request details in comment events
        if "pullRequest" in resource:
            pull_req = resource["pullRequest"]
            repo = pull_req.get("repository", {}).get("webUrl", "")
            pr_id = pull_req.get("pullRequestId")
            if repo and pr_id:
                pr_url = f"{repo}/pullrequest/{pr_id}"
                return unquote(pr_url).replace("pullRequests", "pullrequest")
        
        # Check for standard _links.web.href in PR created/updated events
        if "_links" in resource and "web" in resource["_links"]:
            href = resource["_links"]["web"].get("href", "")
            if href:
                pr_url = unquote(href.replace("_apis/git/repositories", "_git"))
                return pr_url.replace("pullRequests", "pullrequest")
                
    return None

def format_findings(findings: Any) -> str:
    """Formats structured findings list into a readable text block for the LLM."""
    if isinstance(findings, list):
        formatted = ""
        for idx, item in enumerate(findings, 1):
            # Handle both FindingItem object and raw dict
            f = item.dict() if hasattr(item, "dict") else item
            formatted += (
                f"Finding {idx}:\n"
                f"- Summary: {f.get('suggestion_summary', 'N/A')}\n"
                f"- File: {f.get('relevant_file', 'N/A')}\n"
                f"- Lines: {f.get('relevant_lines_start', 'N/A')}-{f.get('relevant_lines_end', 'N/A')}\n"
                f"- Original Score: {f.get('suggestion_score', 'N/A')}\n"
                f"- Why: {f.get('why', 'N/A')}\n"
                f"- Original Severity: {f.get('severity', 'N/A')}\n\n"
            )
        return formatted
    return str(findings)

def format_suggestions_as_yaml(raw_data: Any) -> str:
    """Formats raw PR-Agent suggestions into a clean YAML format as requested."""
    if not raw_data or not isinstance(raw_data, dict):
        return "code_suggestions: []"
    
    suggestions_list = raw_data.get("code_suggestions", [])
    if not suggestions_list:
        return "code_suggestions: []"
        
    yaml_lines = ["code_suggestions:"]
    for s in suggestions_list:
        summary = s.get("one_sentence_summary") or s.get("suggestion_content") or ""
        summary = summary.strip().replace("\r\n", "\n")
        
        file = s.get("relevant_file", "").strip()
        start = s.get("relevant_lines_start", 0)
        end = s.get("relevant_lines_end", 0)
        score = s.get("score", 0)
        
        why = s.get("score_why") or s.get("why") or ""
        why = why.strip().replace("\r\n", "\n")
        
        yaml_lines.append(f"- suggestion_summary: |")
        for line in summary.split("\n"):
            yaml_lines.append(f"    {line}")
            
        yaml_lines.append(f"  relevant_file: \"{file}\"")
        yaml_lines.append(f"  relevant_lines_start: {start}")
        yaml_lines.append(f"  relevant_lines_end: {end}")
        yaml_lines.append(f"  suggestion_score: {score}")
        
        yaml_lines.append(f"  why: |")
        for line in why.split("\n"):
            yaml_lines.append(f"    {line}")
            
    return "\n".join(yaml_lines)

def is_text_similar(text1: str, text2: str, threshold: float = 0.45) -> bool:
    from difflib import SequenceMatcher
    if not text1 or not text2:
        return False
    t1 = text1.strip().lower()
    t2 = text2.strip().lower()
    
    # Check for direct substring matches
    if t1 in t2 or t2 in t1:
        return True
        
    # Check significant word overlap
    words1 = set([w for w in t1.split() if len(w) > 3])
    words2 = set([w for w in t2.split() if len(w) > 3])
    if words1 and words2:
        overlap = words1.intersection(words2)
        if len(overlap) >= 3:
            return True
            
    # SequenceMatcher fallback
    return SequenceMatcher(None, t1, t2).ratio() >= threshold

def deduplicate_suggestions(pr_agent_suggestions_list: list, previous_suggestions: list) -> list:
    """
    Filters out any suggestions from PR-Agent that overlap with the previous agent's findings
    only if they also target similar issues (based on text content and details).
    """
    filtered = []
    for s in pr_agent_suggestions_list:
        file_path = s.get("relevant_file", "").strip().lower()
        start = s.get("relevant_lines_start")
        end = s.get("relevant_lines_end")
        content = (s.get("suggestion_content") or "").strip().lower()
        summary = (s.get("one_sentence_summary") or "").strip().lower()
        
        is_duplicate = False
        for prev in previous_suggestions:
            prev_file = prev.relevant_file.strip().lower() if hasattr(prev, "relevant_file") else prev.get("relevant_file", "").strip().lower()
            
            # Match file paths
            if file_path == prev_file or file_path.endswith(prev_file) or prev_file.endswith(file_path):
                prev_summary = (prev.suggestion_summary if hasattr(prev, "suggestion_summary") else prev.get("suggestion_summary", "")).strip().lower()
                prev_description = (prev.why if hasattr(prev, "why") else prev.get("why", "")).strip().lower()
                
                # Check for line overlap
                prev_start = prev.relevant_lines_start if hasattr(prev, "relevant_lines_start") else prev.get("relevant_lines_start")
                prev_end = prev.relevant_lines_end if hasattr(prev, "relevant_lines_end") else prev.get("relevant_lines_end")
                
                lines_overlap = False
                try:
                    if start is not None and prev_start is not None and str(start).strip() != "None" and str(prev_start).strip() != "None":
                        s_start = int(start)
                        s_end = int(end) if end is not None else s_start
                        p_start = int(prev_start)
                        p_end = int(prev_end) if prev_end is not None else p_start
                        
                        if max(s_start, p_start) <= min(s_end, p_end):
                            lines_overlap = True
                except (ValueError, TypeError):
                    pass
                
                # We consider it a duplicate if:
                # 1. The lines overlap AND the text discusses a similar issue
                # 2. OR the text is highly similar regardless of exact lines
                if lines_overlap:
                    if is_text_similar(summary, prev_summary) or is_text_similar(content, prev_description) or is_text_similar(summary, prev_description):
                        is_duplicate = True
                        break
                else:
                    # High threshold similarity check for non-overlapping lines
                    if is_text_similar(summary, prev_summary, threshold=0.6) or is_text_similar(content, prev_description, threshold=0.6):
                        is_duplicate = True
                        break
                        
        if not is_duplicate:
            filtered.append(s)
            
    return filtered

def post_merged_findings(url: str, payload: dict):
    import requests
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"[INFO] Post to ngrok completed. Status code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"[ERROR] Failed to post to ngrok: {e}")

async def wait_and_run_pr_agent(pr_url: str, pr_number: str):
    """
    Background worker that registers the PR, waits for findings,
    injects them dynamically into the user prompt templates to guide
    the LLM to perform only a gap analysis, and executes the PR-Agent.
    """
    print(f"[INFO] Started integration pipeline for PR: {pr_url} (PR #{pr_number})")
    
    # Store original prompt templates and settings to restore them in finally block
    original_prompts = {}
    for setting_key in ["pr_code_suggestions_prompt", "pr_code_suggestions_prompt_not_decoupled"]:
        prompt_obj = get_settings().get(setting_key, {})
        if prompt_obj and "user" in prompt_obj:
            original_prompts[setting_key] = prompt_obj["user"]
            
    original_focus = get_settings().get("pr_code_suggestions.focus_only_on_problems")
    original_num_suggestions = get_settings().get("pr_code_suggestions.num_code_suggestions_per_chunk")
    original_extra_instructions = get_settings().get("pr_code_suggestions.extra_instructions")
    original_reasoning = get_settings().get("config.model_reasoning")
    
    # Initialize the event in registry using pr_number as the key
    event_info = {"pr_url": pr_url, "event": asyncio.Event(), "findings": None}
    pending_reviews[pr_number] = event_info
    
    # 10-minute timeout for friend's agent to post findings
    timeout_seconds = 600.0
    
    try:
        # Wait non-blockingly until the event is set by /findings
        await asyncio.wait_for(event_info["event"].wait(), timeout=timeout_seconds)
        
        # Format the received findings
        raw_findings = event_info["findings"]
        findings_text = format_findings(raw_findings)
        print(f"[INFO] Findings received for PR #{pr_number}. Running PR-Agent...")
        
        # Optimize PR-Agent configurations to find deep and comprehensive issues that previous agent missed:
        # Disable focus_only_on_problems to allow best practices, performance, and refactoring recommendations.
        get_settings().set("pr_code_suggestions.focus_only_on_problems", False)
        # Increase suggestions per chunk to 3 to generate more detailed options.
        get_settings().set("pr_code_suggestions.num_code_suggestions_per_chunk", 3)
        # Set the reasoning model for the second call (self-reflection) to versatile to split Groq's TPM limits
        get_settings().set("config.model_reasoning", "groq/llama-3.3-70b-versatile")
        
        # Set findings_text as extra_instructions so it is passed as a variable value and escaped from Jinja compilation
        get_settings().set("pr_code_suggestions.extra_instructions", findings_text)
        
        # Dynamically modify both user prompt templates in-memory
        for setting_key in ["pr_code_suggestions_prompt", "pr_code_suggestions_prompt_not_decoupled"]:
            prompt_obj = get_settings().get(setting_key, {})
            if not prompt_obj or "user" not in prompt_obj:
                continue
            
            base_user_prompt = prompt_obj["user"]
            target = "Response (should be a valid YAML, and nothing else):"
            
            gap_analysis_prompt = (
                "\nCRITICAL REQUIREMENT (GAP ANALYSIS):\n"
                "A previous AI agent has already analyzed the PR and made the suggestions listed in "
                "the 'Extra user-provided instructions' section.\n\n"
                "Your job is to identify only ADDITIONAL/NEW bugs, security issues, performance issues, or code smells "
                "that the previous agent MISSED. Focus on deeper code quality, architectural, and performance improvements.\n"
                "- STRICTLY FORBIDDEN: Do not repeat, rephrase, or duplicate any of those suggestions. "
                "If you cannot find any new issues, return an empty list of suggestions: `code_suggestions: []`.\n\n"
            )
            
            if target in base_user_prompt:
                updated_user_prompt = base_user_prompt.replace(
                    target,
                    gap_analysis_prompt + target
                )
            else:
                updated_user_prompt = base_user_prompt + "\n" + gap_analysis_prompt
                
            get_settings().set(f"{setting_key}.user", updated_user_prompt)
        
        # Force publish_output to False to ensure dry-run mode
        get_settings().set("config.publish_output", False)
        
        # Instantiate and run the PR-Agent asynchronously
        agent = PRAgent()
        success = await agent.handle_request(pr_url, "improve")
        print(f"[INFO] PR-Agent pipeline completed for PR #{pr_number}. Success={success}")
        
        # Retrieve the raw structured suggestions data from PR-Agent
        raw_suggestions = get_settings().get("data", {}).get("raw_data", {})
        
        # Apply Python-level deduplication to guarantee zero duplication
        suggestions_list = raw_suggestions.get("code_suggestions", []) if isinstance(raw_suggestions, dict) else []
        print(f"\n[DEBUG] Raw suggestions from LLM before filtering: {len(suggestions_list)}")
        for idx, s in enumerate(suggestions_list, 1):
            print(f"  Raw Suggestion {idx}: {s.get('one_sentence_summary')} (File: {s.get('relevant_file')})")
            
        filtered_suggestions = deduplicate_suggestions(suggestions_list, raw_findings)
        print(f"[DEBUG] Suggestions after filtering: {len(filtered_suggestions)}")
        
        if isinstance(raw_suggestions, dict):
            raw_suggestions["code_suggestions"] = filtered_suggestions
            
        pr_agent_suggestions = format_suggestions_as_yaml(raw_suggestions)
        
        # Print results to the terminal
        print("\n" + "="*80)
        print("                 INTEGRATION SERVER PIPELINE RESULT                     ")
        print("="*80)
        print("\n--- [ AIDER AGENT  SUGGESTIONS] ---")
        print(findings_text)
        print("--- [ PR-AGENT SUGGESTIONS (YAML)] ---")
        print(pr_agent_suggestions)
        print("="*80 + "\n")
        
        # Convert new suggestions to the required format
        new_suggestions = []
        for s in filtered_suggestions:
            score_val = s.get("score")
            confidence = 0.7
            if score_val is not None:
                try:
                    confidence = round(float(score_val) / 10.0, 2)
                except ValueError:
                    pass
                    
            severity = "minor"
            if score_val is not None:
                try:
                    iscore = int(score_val)
                    if iscore >= 9:
                        severity = "critical"
                    elif iscore >= 7:
                        severity = "major"
                    elif iscore >= 4:
                        severity = "medium"
                except ValueError:
                    pass
                    
            label = s.get("label", "code_quality").strip().lower()
            category = "code_quality"
            if "security" in label:
                category = "security"
            elif "performance" in label:
                category = "performance"
            elif "bug" in label or "issue" in label:
                category = "bug"
                
            description = s.get("suggestion_content", "").strip()
            improved_code = s.get("improved_code", "").strip()
            if improved_code:
                description += f"\n\nImproved Code:\n```\n{improved_code}\n```"
                
            new_suggestions.append({
                "file_path": s.get("relevant_file", "").strip(),
                "line_number": s.get("relevant_lines_start"),
                "severity": severity,
                "category": category,
                "description": description,
                "suggestion": s.get("one_sentence_summary", "").strip(),
                "confidence": confidence
            })
            
        # Retrieve original suggestions and merge
        original_suggestions = event_info.get("original_suggestions", [])
        merged_suggestions = original_suggestions + new_suggestions
        
        merged_payload = {
            "pr_id": pr_number,
            "source": "merged-pr-agent",
            "refined_findings": merged_suggestions
        }
        
        ngrok_url = "https://recall-passable-prenatal.ngrok-free.dev/api/findings/submit"
        print(f"[INFO] Posting merged findings to ngrok endpoint: {ngrok_url}")
        await asyncio.to_thread(post_merged_findings, ngrok_url, merged_payload)
        
    except asyncio.TimeoutError:
        print(f"[WARNING] Timed out waiting for findings for PR #{pr_number}")
    except Exception as e:
        print(f"[ERROR] Failed during pipeline execution: {e}")
    finally:
        # Clean up registry
        pending_reviews.pop(pr_number, None)
        # Restore original prompt templates to prevent setting contamination across runs
        for setting_key, original_prompt in original_prompts.items():
            get_settings().set(f"{setting_key}.user", original_prompt)
        # Restore original configuration values
        if original_focus is not None:
            get_settings().set("pr_code_suggestions.focus_only_on_problems", original_focus)
        if original_num_suggestions is not None:
            get_settings().set("pr_code_suggestions.num_code_suggestions_per_chunk", original_num_suggestions)
        if original_extra_instructions is not None:
            get_settings().set("pr_code_suggestions.extra_instructions", original_extra_instructions)
        if original_reasoning is not None:
            get_settings().set("config.model_reasoning", original_reasoning)

@app.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receives incoming webhook notifications. Launches the background waiting task
    and immediately returns 200 OK.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    pr_url = extract_pr_url(body)
    if not pr_url:
        raise HTTPException(status_code=400, detail="PR URL not found in request payload")
        
    pr_number = extract_pr_number(pr_url)
    
    # Start the waiting worker in the background
    background_tasks.add_task(wait_and_run_pr_agent, pr_url, pr_number)
    return {"status": "waiting_for_findings", "pr_url": pr_url, "pr_number": pr_number}

@app.post("/api/pr-agent/submit")
async def receive_findings(payload: IncomingFindingsPayload, background_tasks: BackgroundTasks):
    """
    Receives findings/suggestions from the external agent,
    translates them to internal format, and wakes up the background wait task.
    """
    pr_id_str = str(payload.pr_id)
    pr_number = extract_pr_number(pr_id_str)
    
    # If the PR is not registered, but we have a full PR URL, register it on the fly
    if pr_number not in pending_reviews:
        if pr_id_str.startswith("http") and ("/pull/" in pr_id_str or "/pullrequest/" in pr_id_str):
            print(f"[INFO] PR #{pr_number} not registered but valid URL received. Registering on-the-fly...")
            event_info = {"pr_url": pr_id_str, "event": asyncio.Event(), "findings": None}
            pending_reviews[pr_number] = event_info
            background_tasks.add_task(wait_and_run_pr_agent, pr_id_str, pr_number)
        else:
            raise HTTPException(status_code=404, detail=f"PR #{pr_number} not registered or timed out")
        
    translated_findings = []
    for item in payload.my_suggestions:
        # Prepend category to description if present
        why_text = item.description or ""
        if item.category:
            why_text = f"[{item.category}] {why_text}"
            
        # Convert confidence (0-1) to suggestion_score (1-10)
        score = None
        if item.confidence is not None:
            try:
                score = int(float(item.confidence) * 10)
            except Exception:
                pass
                
        finding = FindingItem(
            suggestion_summary=item.suggestion,
            relevant_file=item.file_path,
            relevant_lines_start=item.line_number,
            # relevant_lines_end=item.line_number,
            suggestion_score=score,
            why=why_text,
            severity=item.severity
        )
        translated_findings.append(finding)
        
    # Store findings and set event to trigger wait_and_run_pr_agent
    pending_reviews[pr_number]["findings"] = translated_findings
    pending_reviews[pr_number]["original_suggestions"] = [item.dict() for item in payload.my_suggestions]
    pending_reviews[pr_number]["event"].set()
    
    return {"status": "accepted", "pr_number": pr_number}

if __name__ == "__main__":
    uvicorn.run("pr_agent.app17:app", host="0.0.0.0", port=8000, reload=True)
