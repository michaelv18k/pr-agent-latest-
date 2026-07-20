import asyncio
import os
import re
from typing import Any, Dict, List, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pr_agent.agent.pr_agent import PRAgent
from pr_agent.config_loader import get_settings

# ---------------------------------------------------------------------------
# ADO config will be read per-request to allow dotenv or uvicorn to load 
# env vars dynamically.
# ---------------------------------------------------------------------------

app = FastAPI(title="PR-Agent & Aider-Agent Integration Server")

# Serializes concurrent requests so no two PRs mutate global settings at the same time.
# PR-Agent uses a global settings singleton — without this lock, two simultaneous
# PR reviews would overwrite each other's injected findings mid-flight.
_settings_lock = asyncio.Lock()

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

class FindingItem(BaseModel):
    suggestion_summary: str = None
    relevant_file: str = None
    relevant_lines_start: Union[int, str, None] = None
    relevant_lines_end: Union[int, str, None] = None
    suggestion_score: Union[int, str, None] = None
    why: str = None
    severity: str = None

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
    """Extracts the PR number/id from a GitHub or Azure DevOps PR URL."""
    match = re.search(r'/pull/(\d+)', pr_url)
    if match:
        return match.group(1)
    match = re.search(r'/pullrequest/(\d+)', pr_url, re.IGNORECASE)
    if match:
        return match.group(1)
    return pr_url

def format_findings(findings: Any) -> str:
    """Formats structured findings list into a readable text block for the LLM."""
    if isinstance(findings, list):
        formatted = ""
        for idx, item in enumerate(findings, 1):
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

def is_text_similar(text1: str, text2: str, threshold: float = 0.45) -> bool:
    from difflib import SequenceMatcher
    if not text1 or not text2:
        return False
    t1 = text1.strip().lower()
    t2 = text2.strip().lower()
    
    if t1 in t2 or t2 in t1:
        return True
        
    words1 = set([w for w in t1.split() if len(w) > 3])
    words2 = set([w for w in t2.split() if len(w) > 3])
    if words1 and words2:
        overlap = words1.intersection(words2)
        if len(overlap) >= 3:
            return True
            
    return SequenceMatcher(None, t1, t2).ratio() >= threshold

def deduplicate_suggestions(pr_agent_suggestions_list: list, previous_suggestions: list) -> list:
    """Filters out overlapping suggestions."""
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
            
            if file_path == prev_file or file_path.endswith(prev_file) or prev_file.endswith(file_path):
                prev_summary = (prev.suggestion_summary if hasattr(prev, "suggestion_summary") else prev.get("suggestion_summary", "")).strip().lower()
                prev_description = (prev.why if hasattr(prev, "why") else prev.get("why", "")).strip().lower()
                
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
                
                if lines_overlap:
                    if is_text_similar(summary, prev_summary) or is_text_similar(content, prev_description) or is_text_similar(summary, prev_description):
                        is_duplicate = True
                        break
                else:
                    if is_text_similar(summary, prev_summary, threshold=0.6) or is_text_similar(content, prev_description, threshold=0.6):
                        is_duplicate = True
                        break
                        
        if not is_duplicate:
            filtered.append(s)
            
    return filtered

@app.post("/api/pr-agent/submit")
async def receive_findings(payload: IncomingFindingsPayload):
    """
    Receives findings from Azure DevOps pipeline, runs PR-Agent synchronously,
    and returns the merged findings directly in the HTTP response.
    """
    pr_url = str(payload.pr_id)
    
    # Auto-format as ADO URL if only a PR ID was passed
    if pr_url.isdigit():
        _ADO_ORG     = os.environ.get("AZURE_DEVOPS_ORG")
        _ADO_PROJECT = os.environ.get("AZURE_DEVOPS_PROJECT")
        _ADO_REPO    = os.environ.get("AZURE_DEVOPS_REPO")
        
        _missing = [k for k, v in {
            "AZURE_DEVOPS_ORG":     _ADO_ORG,
            "AZURE_DEVOPS_PROJECT": _ADO_PROJECT,
            "AZURE_DEVOPS_REPO":    _ADO_REPO,
        }.items() if not v]
        
        if _missing:
            raise HTTPException(
                status_code=500, 
                detail=f"[app18] Missing required environment variable(s): {', '.join(_missing)}"
            )

        pr_url = (
            f"https://dev.azure.com/{_ADO_ORG}/{_ADO_PROJECT}"
            f"/_git/{_ADO_REPO}/pullrequest/{pr_url}"
        )

    pr_number = extract_pr_number(pr_url)
    
    print(f"[INFO] Started synchronous pipeline for PR: {pr_url} (PR #{pr_number})")
    
    # 1. Translate incoming findings
    translated_findings = []
    for item in payload.my_suggestions:
        why_text = item.description or ""
        if item.category:
            why_text = f"[{item.category}] {why_text}"
            
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
            suggestion_score=score,
            why=why_text,
            severity=item.severity
        )
        translated_findings.append(finding)
        
    findings_text = format_findings(translated_findings)
    original_suggestions = [item.dict() for item in payload.my_suggestions]

    # Acquire lock before touching any global settings.
    # This serializes concurrent PR reviews so they don't overwrite each other.
    async with _settings_lock:
        # Store original prompt templates to restore in finally block
        original_prompts = {}
        for setting_key in ["pr_code_suggestions_prompt", "pr_code_suggestions_prompt_not_decoupled"]:
            prompt_obj = get_settings().get(setting_key, {})
            if prompt_obj and "user" in prompt_obj:
                original_prompts[setting_key] = prompt_obj["user"]

        original_focus = get_settings().get("pr_code_suggestions.focus_only_on_problems")
        original_num_suggestions = get_settings().get("pr_code_suggestions.num_code_suggestions_per_chunk")
        original_extra_instructions = get_settings().get("pr_code_suggestions.extra_instructions")
        original_reasoning = get_settings().get("config.model_reasoning")

        try:
            # Optimize PR-Agent configurations
            get_settings().set("pr_code_suggestions.focus_only_on_problems", False)
            get_settings().set("pr_code_suggestions.num_code_suggestions_per_chunk", 3)
            get_settings().set("config.model_reasoning", "groq/llama-3.3-70b-versatile")
            get_settings().set("pr_code_suggestions.extra_instructions", findings_text)

            # Dynamically modify user prompt templates
            for setting_key in ["pr_code_suggestions_prompt", "pr_code_suggestions_prompt_not_decoupled"]:
                prompt_obj = get_settings().get(setting_key, {})
                if not prompt_obj or "user" not in prompt_obj:
                    continue

                base_user_prompt = prompt_obj["user"]
                target = "Response (should be a valid YAML, and nothing else):"
                verification_prompt = (
                    "\nCRITICAL REQUIREMENT (VERIFICATION & GAP ANALYSIS):\n"
                    "A previous AI agent has analyzed the PR and made the suggestions listed in "
                    "the 'Extra user-provided instructions' section.\n\n"
                    "Your job has two parts:\n"
                    "1. VERIFY: Review each of the previous agent's suggestions. If a suggestion is correct and highly valuable, INCLUDE it in your output (you may improve the explanation or code). If it is a false positive, hallucination, or low-value, DO NOT include it.\n"
                    "2. DISCOVER: Identify any ADDITIONAL/NEW bugs, security issues, or performance problems that the previous agent missed.\n\n"
                    "STRICT QUALITY STANDARDS (You MUST drop any suggestion that violates these):\n"
                    "- DO NOT suggest adding hardcoded secrets, passwords, or fallback API keys. Missing secrets must be handled via secure exceptions.\n"
                    "- DO NOT suggest adding `# noqa` to suppress unused imports; the correct suggestion is to delete the unused import.\n"
                    "- FALSE POSITIVE OVERRIDE: If you inspect the code and realize it ALREADY implements the suggestion perfectly (e.g. it is already parameterized) and your `improved_code` would be identical to the `existing_code`, you MUST DROP the suggestion entirely. This overrides the confidence rule below.\n"
                    "- CONFIDENCE PRESERVATION: If a suggestion has an 'Original Score' of 9 or 10 (and is not a false positive), DO NOT hedge your bets. You MUST output a `suggestion_score` of 9 or 10 for it.\n"
                    "- DIFF PARSING RULE: When extracting `existing_code`, you MUST ONLY extract the added/current lines (lines starting with `+` or space in the diff). NEVER extract deleted lines (lines starting with `-`).\n\n"
                    "- Output a single, comprehensive list of `code_suggestions` containing both the verified previous suggestions and your new discoveries.\n\n"
                )

                if target in base_user_prompt:
                    updated_user_prompt = base_user_prompt.replace(target, verification_prompt + target)
                else:
                    updated_user_prompt = base_user_prompt + "\n" + verification_prompt
                get_settings().set(f"{setting_key}.user", updated_user_prompt)

            get_settings().set("config.publish_output", False)

            # 2. RUN PR-AGENT SYNCHRONOUSLY
            agent = PRAgent()
            success = await agent.handle_request(pr_url, "improve")
            print(f"[INFO] PR-Agent pipeline completed. Success={success}")

            raw_suggestions = get_settings().get("data", {}).get("raw_data", {})
            suggestions_list = raw_suggestions.get("code_suggestions", []) if isinstance(raw_suggestions, dict) else []

            # We no longer deduplicate against the original findings, because the LLM is expected
            # to output the valid original findings along with any new ones.
            filtered_suggestions = suggestions_list

            # 3. Format back to standard JSON
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
                            severity = "major"  # mapped from medium so Aider picks it up
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

                description = (s.get("suggestion_content") or s.get("why") or "").strip()
                improved_code = s.get("improved_code", "").strip()
                if improved_code:
                    description += f"\n\nImproved Code:\n```\n{improved_code}\n```"

                suggestion_title = (s.get("one_sentence_summary") or s.get("suggestion_summary") or "").strip()

                new_suggestions.append({
                    "file_path": s.get("relevant_file", "").strip(),
                    "line_number": s.get("relevant_lines_start"),
                    "severity": severity,
                    "category": category,
                    "description": description,
                    "suggestion": suggestion_title,
                    "confidence": confidence
                })

            # The new_suggestions list now contains the fully verified and augmented findings
            merged_suggestions = new_suggestions

            # 4. RETURN DIRECTLY TO CALLER (NO WEBHOOK!)
            return {
                "status": "success",
                "pr_number": pr_number,
                "refined_findings": merged_suggestions
            }

        except Exception as e:
            print(f"[ERROR] Failed during pipeline execution: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Restore all settings — guaranteed to run before the next request acquires the lock
            for setting_key, original_prompt in original_prompts.items():
                get_settings().set(f"{setting_key}.user", original_prompt)
            if original_focus is not None:
                get_settings().set("pr_code_suggestions.focus_only_on_problems", original_focus)
            if original_num_suggestions is not None:
                get_settings().set("pr_code_suggestions.num_code_suggestions_per_chunk", original_num_suggestions)
            if original_extra_instructions is not None:
                get_settings().set("pr_code_suggestions.extra_instructions", original_extra_instructions)
            if original_reasoning is not None:
                get_settings().set("config.model_reasoning", original_reasoning)


if __name__ == "__main__":
    uvicorn.run("pr_agent.app18:app", host="0.0.0.0", port=8000, reload=True)

