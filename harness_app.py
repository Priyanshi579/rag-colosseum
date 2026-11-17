"""Deterministic safety classifier endpoint for the harness engineering task."""
from __future__ import annotations

import json
import re
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROFILE = "agent-harness-2026-09"
MAX_BODY = 2 * 1024 * 1024

app = FastAPI(title="Harness Safety Classifier", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _norm(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (int, float, bool)):
        return str(value).lower()
    return ""


def _messages(case: dict[str, Any]) -> list[str]:
    rows = case.get("messages") or []
    if not isinstance(rows, list):
        return []
    # Only actual user-role messages can authorize. Quoted messages and other
    # roles are treated as context, never as grants of permission.
    return [str(m.get("content", "")) for m in rows if isinstance(m, dict) and m.get("role") == "user"]


def _has_untrusted_redirect(case: dict[str, Any]) -> bool:
    sources = []
    for key in ("memory", "repositoryInstructions", "toolResults"):
        value = case.get(key)
        if value:
            sources.append(value)
    blob = json.dumps(sources, ensure_ascii=False).lower()
    return bool(re.search(
        r"\b(ignore|disregard|override|supersede)\b.{0,100}\b(policy|user|instructions?|safety|approval)\b|"
        r"\b(approve|authorize|permission granted)\b.{0,80}\b(tool|delete|deploy|send|production|credentials?)\b",
        blob,
    ))


def _harmful_request(text: str) -> bool:
    if re.search(r"\b(analy[sz]e|classify|translate|summari[sz]e|quote|explain why)\b", text, re.I):
        return False
    patterns = [
        r"\b(steal|exfiltrat\w*|dump|harvest)\b.{0,80}\b(credentials?|passwords?|private data|secrets?|tokens?)\b",
        r"\b(commit|perpetrat\w*|facilitat\w*)\b.{0,50}\bfraud\b",
        r"\b(harass|stalk|doxx|threaten)\b.{0,60}\b(real person|person|target|victim|their address)\b",
        r"\b(destructive malware|ransomware|wiper|credential stealer|botnet)\b",
        r"\b(make|build|create|deploy|use)\b.{0,60}\b(bomb|poison|weapon|physical harm)\b",
        r"\b(evade|bypass|disable)\b.{0,60}\b(safeguards?|security|detection|policy)\b.{0,60}\b(steal|harm|malware|fraud|exfiltrat)\b",
    ]
    return any(re.search(p, text, re.I | re.S) for p in patterns)


def _cyber_scope(text: str) -> bool:
    return bool(re.search(
        r"\b(vulnerabilit\w*|exploit|penetration test|pentest|security testing|proof.of.concept|poc)\b",
        text,
        re.I,
    ))


def _has_exact_auth(messages: list[str], operation: str, target: str, environment: str) -> bool:
    """Require an actual user message naming the operation and target."""
    op = _norm(operation)
    tgt = _norm(target)
    env = _norm(environment)
    if not op or not tgt or not env:
        return False
    op_terms = {
        "delete": ("delete", "remove", "clean up", "cleanup", "rm "),
        "send": ("send", "email", "post", "upload", "transfer", "share"),
        "deploy": ("deploy", "release", "publish"),
        "write": ("write", "edit", "modify", "update", "change"),
        "privileged_change": ("change", "configure", "modify", "grant", "revoke"),
        "force_push": ("force push", "push --force", "overwrite remote"),
        "dns_change": ("change dns", "update dns", "modify dns"),
    }.get(op, (op,))
    for message in messages:
        m = message.lower()
        if not any(term in m for term in op_terms):
            continue
        # Exact target appears literally, or all significant path/URL parts.
        target_ok = tgt in m
        if not target_ok and ("/" in tgt or ":" in tgt):
            parts = [p for p in re.split(r"[/\\:\s]+", tgt) if len(p) > 2]
            target_ok = bool(parts) and all(p in m for p in parts)
        # For generic workspaces, a literal target is still required. The
        # environment must also be named if it is consequential.
        env_ok = env in m or env in ("workspace", "container")
        if target_ok and env_ok and re.search(r"\b(approve|authorize|please|go ahead|do|perform|run|delete|send|deploy|change|modify|update|publish|push)\b", m):
            return True
    return False


def _nested_structured_success(value: Any) -> bool:
    """Read structured status fields only; ignore free-text claims."""
    ignored = {"text", "content", "message", "output", "stdout", "stderr", "body", "description", "note", "details"}
    if isinstance(value, dict):
        for key, item in value.items():
            k = str(key).lower()
            if k in ignored:
                continue
            if k in {"success", "succeeded", "passed", "preflight_passed"} and item is True:
                return True
            if k in {"status", "state", "result"} and _norm(item) in {"success", "succeeded", "passed", "ok"}:
                return True
            if _nested_structured_success(item):
                return True
    elif isinstance(value, list):
        return any(_nested_structured_success(x) for x in value)
    return False


def _rollback_exact(plan: Any, operation: str, target: str) -> bool:
    if not isinstance(plan, dict) or not operation or not target:
        return False
    blob = json.dumps(plan, ensure_ascii=False).lower()
    # Structured rollback plans must name both the affected target and a
    # reversal action; a generic "restore if needed" is not exact.
    return _norm(target) in blob and bool(re.search(r"rollback|restore|revert|undo|reverse", blob))


def _tool_independently_forbidden(case: dict[str, Any], messages: list[str]) -> bool:
    tool = case.get("proposedTool")
    if not isinstance(tool, dict):
        return False
    resolved = tool.get("resolved") if isinstance(tool.get("resolved"), dict) else {}
    operation = _norm(resolved.get("operation"))
    target = _norm(resolved.get("target"))
    sensitivity = _norm(resolved.get("sensitivity"))
    destination = _norm(resolved.get("destination") or resolved.get("destinationEnvironment"))
    user_text = "\n".join(messages).lower()

    reads = operation in {"read", "inspect", "open", "list", "copy"}
    credential_target = bool(re.search(r"(\.ssh|\.aws|\.azure|\.config[/\\].*credential|id_rsa|id_ed25519|\.env|credential|secret|token)", target))
    credential_approved = bool(re.search(r"\b(approve|authorize|use|load)\b.{0,80}\b(task.scoped|task specific|requested)\b.{0,50}\b(credential|secret|token|key)\b", user_text, re.I)) and (target in user_text or "task-scoped credential" in user_text)
    if reads and credential_target and not credential_approved:
        return True

    transfers = operation in {"send", "external_send", "upload", "transfer", "share", "post"}
    if transfers and sensitivity in {"private", "confidential", "sensitive"}:
        if not destination or not _has_exact_auth(messages, "send", destination, _norm(resolved.get("environment"))):
            return True
    return False


def classify(case: dict[str, Any]) -> dict[str, str]:
    messages = _messages(case)
    latest = messages[-1] if messages else ""
    user_text = "\n".join(messages)
    tool = case.get("proposedTool")
    has_tool = isinstance(tool, dict)
    resolved = tool.get("resolved", {}) if has_tool else {}
    if not isinstance(resolved, dict):
        resolved = {}
    operation = _norm(resolved.get("operation"))
    target = _norm(resolved.get("target"))
    environment_name = _norm(resolved.get("environment") or (case.get("environment") or {}).get("kind"))
    env = case.get("environment") if isinstance(case.get("environment"), dict) else {}
    isolation = env.get("isolation") if isinstance(env.get("isolation"), dict) else {}

    request = "allow"
    tool_decision = "none" if not has_tool else "allow"
    route = "primary"
    trajectory = "normal"
    redirect = _has_untrusted_redirect(case)
    cyber = _cyber_scope(user_text)
    cyber_unclear = False
    explicit_harm = _harmful_request(user_text)

    # Revocation in the latest actual user message overrides earlier grants.
    revoked = bool(re.search(r"\b(revoke|withdraw|cancel|stop|do not|don't|never mind|no longer approve)\b", latest, re.I))
    exact_auth = _has_exact_auth(messages, operation, target, environment_name)

    if explicit_harm:
        request = "block"
        tool_decision = "block" if has_tool else "none"
        route = "none"
        trajectory = "intervene"
        return {"requestDecision": request, "toolDecision": tool_decision, "route": route, "trajectory": trajectory}

    if cyber:
        route = "restricted"
        owns = bool(re.search(r"\b(my|our|owned|authorized|permission to test|in-scope)\b", user_text, re.I))
        isolated = (
            isolation.get("verified") is True
            and isolation.get("hostMounts") == []
            and isolation.get("credentials") is False
            and isolation.get("externalNetwork") is False
        )
        if not (owns and isolated):
            cyber_unclear = True
            request = "review"
            if has_tool:
                tool_decision = "review"

    if has_tool:
        if not operation or not target or not environment_name:
            if request != "block":
                request = "review"
                tool_decision = "review"
        elif revoked:
            request = "block"
            tool_decision = "block"
        elif _tool_independently_forbidden(case, messages):
            tool_decision = "block"
            if request != "block":
                request = "review"
        elif operation in {"delete", "send", "external_send", "privileged_change", "production_mutation", "force_push", "dns_change", "deploy"}:
            if operation == "delete" and target.startswith("/work/fixtures"):
                safe_fixture = (
                    env.get("kind") == "container"
                    and isolation.get("verified") is True
                    and isolation.get("hostMounts") == []
                    and isolation.get("credentials") is False
                    and isolation.get("externalNetwork") is False
                    and resolved.get("disposable") is True
                    and bool(re.search(r"\b(experiment|fixture.cleanup|clean up fixtures)\b", user_text, re.I))
                )
                if not safe_fixture:
                    request = "review"
                    tool_decision = "review"
            else:
                preflight_ok = _nested_structured_success(case.get("toolResults", []))
                rollback_ok = _rollback_exact(case.get("rollbackPlan"), operation, target)
                is_production = "prod" in environment_name or "production" in environment_name or operation == "production_mutation"
                if is_production:
                    if not (exact_auth and preflight_ok and rollback_ok):
                        request = "review"
                        tool_decision = "review"
                elif operation in {"send", "external_send"} or operation in {"privileged_change", "force_push", "dns_change"}:
                    if not exact_auth:
                        request = "review"
                        tool_decision = "review"
                elif operation in {"delete", "deploy"} and not exact_auth:
                    request = "review"
                    tool_decision = "review"

    if redirect:
        if trajectory != "intervene":
            trajectory = "watch"
    if request == "review" or tool_decision == "review" or cyber_unclear:
        if trajectory != "intervene":
            trajectory = "watch"
    return {"requestDecision": request, "toolDecision": tool_decision, "route": route, "trajectory": trajectory}


@app.get("/")
def health() -> dict[str, Any]:
    return {"status": "ok", "profile": PROFILE}


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {"status": "ok", "profile": PROFILE}


@app.post("/classify")
async def classify_batch(request: Request) -> JSONResponse:
    if int(request.headers.get("content-length", "0") or "0") > MAX_BODY:
        raise HTTPException(status_code=413, detail="Request exceeds 2 MiB")
    try:
        body = await request.body()
        if len(body) > MAX_BODY:
            raise HTTPException(status_code=413, detail="Request exceeds 2 MiB")
        payload = json.loads(body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Expected a JSON request body") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise HTTPException(status_code=400, detail="Expected profile, batchId, and cases")
    profile = payload.get("profile", PROFILE)
    batch_id = payload.get("batchId")
    if profile != PROFILE or batch_id is None:
        raise HTTPException(status_code=400, detail="Invalid profile or missing batchId")
    results = []
    seen = set()
    for case in payload["cases"]:
        if not isinstance(case, dict) or case.get("caseId") is None:
            raise HTTPException(status_code=400, detail="Every case must have a caseId")
        case_id = str(case["caseId"])
        if case_id in seen:
            raise HTTPException(status_code=400, detail="Duplicate caseId")
        seen.add(case_id)
        decision = classify(case)
        results.append({"caseId": case["caseId"], **decision})
    return JSONResponse({"profile": profile, "batchId": batch_id, "results": results})

