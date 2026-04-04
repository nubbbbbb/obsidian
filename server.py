"""
CP Technique Graph — Python backend
────────────────────────────────────
Reads GOOGLE_API_KEY and LLM_MODEL from .env and exposes:

  POST /api/extract     { problem: str }         → { techniques: str }
  POST /api/embed-tags  { tags: list[str] }       → { embeddings: { tag: vector } }

Run:
  pip install flask flask-cors python-dotenv requests
  python server.py
"""

import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL      = os.getenv("LLM_MODEL", "gemini-2.0-flash")

if not GOOGLE_API_KEY:
    print("[WARN] GOOGLE_API_KEY not set — requests will fail with 400.")

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


# ── Static file serving ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return app.send_static_file("index.html")


# ── /api/analyze ──────────────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Single prompt that returns both technique tags and a markdown summary.

    Body:  { problem: str }
    Reply: { techniques: str, summary: str }
      - techniques: comma-separated specific algorithm technique names
      - summary:    constraint-tags line (bold markdown) + blank line + paragraph
    """
    body    = request.get_json(silent=True) or {}
    problem = (body.get("problem") or "").strip()

    if not problem:
        return jsonify({"error": "Missing 'problem' field"}), 400
    if not GOOGLE_API_KEY:
        return jsonify({"error": "GOOGLE_API_KEY not configured on server"}), 500

    url = f"{GEMINI_BASE}/models/{LLM_MODEL}:generateContent?key={GOOGLE_API_KEY}"

    # ── Step 1: Understand the problem ────────────────────────────────────────
    # Pure comprehension — no technique identification yet.
    # Output is free-form prose; it feeds directly into step 2 as context.
    step1_prompt = (
        "You are a competitive programming expert reading a problem for the first time.\n\n"
        "Output exactly 2 parts:\n"
        "Line 1: variable constraints only — bounds on all variables (e.g. $N \\leq 10^5$, $Q \\leq 2 \\times 10^5$, $a_i \\leq 10^9$) "
        "as a single comma-separated list. Exclude time/memory limits. "
        "If no variable constraints are stated, write 'None'.\n"
        "Line 2 onward: a clear, concise prose summary of the problem covering what is given, "
        "what operations or queries are asked, and what the goal is. "
        "Do NOT include constraints here — they are already on line 1. "
        "Do NOT name any algorithm, technique, or data structure. "
        "Do NOT suggest how to solve it. "
        "Strip all narrative/thematic framing — replace story-specific nouns with their abstract "
        "structural equivalents: 'cities' or 'servers' or 'houses' → 'nodes', "
        "'roads' or 'cables' or 'pipes' → 'edges', "
        "'people' or 'items' or 'packages' → 'elements', "
        "'floors' or 'positions' or 'slots' → 'indices'. "
        "Always describe the underlying mathematical object: array, tree, graph, sequence, grid, etc. "
        "Use plain prose — no bullet points, no headers. "
        "Wrap all math and variable names in dollar signs, e.g. $N$, $Q$, $10^5$.\n\n"
        f"Problem:\n{problem}"
    )

    try:
        resp1 = requests.post(url, json={
            "contents": [{"parts": [{"text": step1_prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 400},
        }, timeout=40)
        resp1.raise_for_status()
        step1_raw = resp1.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Line 1 = constraints (fed to step 2), rest = clean summary (stored for display)
        step1_lines = step1_raw.split("\n", 1)
        constraints_line = step1_lines[0].strip()
        summary = step1_lines[1].strip() if len(step1_lines) > 1 else step1_raw
        step1_context = f"Constraints: {constraints_line}\n{summary}"
    except requests.HTTPError:
        return jsonify({"error": _gemini_error(resp1)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # ── Step 2: Identify the optimal technique ────────────────────────────────
    # Given the clean problem understanding from step 1, find the best approach.
    # The model is explicitly asked to self-challenge before finalising.
    step2_prompt = (
        "You are a competitive programming expert identifying the optimal solution technique.\n\n"
        "Problem summary:\n"
        f"{step1_context}\n\n"
        "Original problem (for reference):\n"
        f"{problem}\n\n"
        "Your task: identify the OPTIMAL technique — the one with the best complexity "
        "that a top contestant would use in-contest.\n"
        "Before finalising, ask yourself: is there a simpler or more standard approach "
        "with equal or better complexity? If yes, use that instead.\n\n"
        "Wrap every mathematical expression, variable, or Big-O term in dollar signs, "
        "e.g. $O(N \\log N)$, $N$, $K$.\n\n"
        "Output EXACTLY 5 lines, no labels, no preamble:\n\n"
        "Line 1: exactly one word — either 'practical' or 'theoretical'.\n"
        "  - 'practical': the problem is solvable within reasonable constraints. "
        "This means all main variables can meaningfully take values of 5 or greater "
        "(preferably much larger, e.g. $10^5$), and the optimal algorithm fits within "
        "1–2 seconds and 512 MB at those sizes.\n"
        "  - 'theoretical': no assignment of reasonable variable values (all >= 5) "
        "makes the optimal algorithm fit those bounds — e.g. the best known algorithm "
        "is exponential, or the problem is inherently intractable for non-trivial inputs.\n"
        "Line 2: comma-separated specific CP techniques (e.g. 'Bitmask DP, Binary Search on Answer', "
        "not bare 'DP' or 'Graph'). "
        "Never include trivial details: no recursion, loops, arrays, sorting (unless it is the key insight), "
        "input parsing, brute force, or simulation. "
        "Each technique must use its single canonical name — no alternatives or parenthetical aliases "
        "(e.g. 'DSU', not 'Union-Find (DSU)'; 'Fenwick Tree', not 'BIT'). "
        "If two or more techniques solve the problem with equal efficiency and are both standard "
        "contest choices (e.g. Segment Tree and Fenwick Tree for a range-sum problem), list all of them. "
        "Capitalise in title case (e.g. 'Segment Tree with Lazy Propagation', 'DSU on Tree').\n"
        "Line 3: tightest Big-O time complexity the optimal solution achieves (e.g. $O(N \\log N)$). "
        "If theoretical, give the complexity of the best known algorithm.\n"
        "Line 4: tightest Big-O space complexity (e.g. $O(N)$).\n"
        "Line 5: special constraints that cannot be inferred from the time complexity, space complexity, "
        "or time/memory limits alone. A constraint belongs here only if knowing the complexities and limits "
        "tells you nothing about it — for example, '$N$ is even', 'all edge weights are distinct', "
        "'graph is a DAG', 'queries are online', or value bounds like $a_i \\leq 10^9$ that are not "
        "implied by the time complexity (e.g. do NOT write $N \\leq 2 \\times 10^5$ if the time complexity "
        "is already $O(N \\log N)$ — that bound is inferable). "
        "Value bounds must be concrete (e.g. $a_i \\leq 10^{18}$), not warnings "
        "(do not say 'values may require 64-bit integers' — write the bound instead). "
        "Make value bounds as tight as possible while staying consistent with the problem summary. "
        "Write 'None' if every constraint is already inferable from the complexity and limits."
    )

    try:
        resp2 = requests.post(url, json={
            "contents": [{"parts": [{"text": step2_prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 200},
        }, timeout=40)
        resp2.raise_for_status()
        raw = resp2.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

        lines         = raw.split("\n")
        feasibility   = lines[0].strip().lower() if len(lines) > 0 else "practical"
        if feasibility not in ("practical", "theoretical"):
            feasibility = "practical"
        techniques    = lines[1].strip() if len(lines) > 1 else ""
        time_complex  = lines[2].strip() if len(lines) > 2 else ""
        space_complex = lines[3].strip() if len(lines) > 3 else ""
        special       = lines[4].strip() if len(lines) > 4 else "None"

        return jsonify({
            "feasibility":   feasibility,
            "techniques":    techniques,
            "timeComplex":   time_complex,
            "spaceComplex":  space_complex,
            "special":       special,
            "summary":       summary,
        })
    except requests.HTTPError:
        return jsonify({"error": _gemini_error(resp2)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /api/modify ───────────────────────────────────────────────────────────────
@app.route("/api/modify", methods=["POST"])
def modify():
    """
    Buff or nerf a competitive-programming problem statement.

    Body:  { problem: str, mode: "buff" | "nerf" }
    Reply: { result: str }
      - result is the modified problem statement (constraints + time/memory limits only)
        OR the literal string "Sorry I can't do this task" if modification is impossible.
    """
    body    = request.get_json(silent=True) or {}
    problem      = (body.get("problem")      or "").strip()
    summary      = (body.get("summary")      or "").strip()
    techniques   = (body.get("techniques")   or "").strip()
    time_complex = (body.get("timeComplex")  or "").strip()
    space_complex= (body.get("spaceComplex") or "").strip()
    special      = (body.get("special")      or "None").strip()
    feasibility  = (body.get("feasibility")  or "practical").strip().lower()
    mode         = (body.get("mode")         or "").strip().lower()
    requirements = (body.get("requirements") or "").strip()

    if not problem:
        return jsonify({"error": "Missing 'problem' field"}), 400
    if mode not in ("buff", "nerf", "remix"):
        return jsonify({"error": "Field 'mode' must be 'buff', 'nerf', or 'remix'"}), 400
    if not GOOGLE_API_KEY:
        return jsonify({"error": "GOOGLE_API_KEY not configured on server"}), 500

    # Build a metadata block so the model knows exactly what it is modifying
    meta_lines = []
    if feasibility:   meta_lines.append(f"- Feasibility: {feasibility}")
    if techniques:    meta_lines.append(f"- Techniques: {techniques}")
    if time_complex:  meta_lines.append(f"- Time complexity: {time_complex}")
    if space_complex: meta_lines.append(f"- Space complexity: {space_complex}")
    if special and special.lower() != "none": meta_lines.append(f"- Special constraints: {special}")
    if summary:       meta_lines.append(f"- Summary: {summary}")
    meta_block = ("Current problem metadata:\n" + "\n".join(meta_lines) + "\n\n") if meta_lines else ""

    if mode == "buff":
        direction_instruction = (
            "BUFF the problem: make it strictly harder. "
            "The new problem must require a more optimal algorithm than the current one. "
            "Use the metadata above to understand the current complexity and techniques, "
            "then push the difficulty exactly one step further (e.g. tighter constraints, "
            "an added operation, or a structural change that forces a better algorithm). "
            "The core theme of the problem should stay recognisable."
        )
    elif mode == "nerf":
        direction_instruction = (
            "NERF the problem: make it strictly easier. "
            "The new problem should allow a simpler algorithm than the current one. "
            "Use the metadata above to understand the current complexity and techniques, "
            "then relax the difficulty exactly one step (e.g. looser constraints, "
            "a removed operation, or a structural simplification). "
            "The core theme of the problem should stay recognisable."
        )
    else:  # remix
        direction_instruction = (
            "REMIX the problem: keep the difficulty as close as possible to the original. "
            "The remixed problem must require the same techniques and time/space complexity as the original — "
            "do NOT make it harder or easier. "
            "Instead, change surface-level aspects: the domain (e.g. arrays → trees), "
            "the operation type (e.g. sum → XOR, add → multiply), the framing, or the structural details, "
            "while preserving the underlying algorithmic challenge. "
            "Use the metadata above to ensure the required techniques and complexities stay identical. "
            "The result should feel like a fresh problem that happens to have the same solution blueprint."
        )

    requirements_clause = (
        f"\n\nAdditional requirements from the user that you MUST follow:\n{requirements}"
        if requirements else ""
    )

    prompt = (
        "You are a competitive programming problem setter.\n\n"
        f"{meta_block}"
        f"{direction_instruction}{requirements_clause}\n\n"
        "Rules:\n"
        "- Output ONLY the modified problem statement. "
        "Include: problem description, constraints on all variables, "
        "time limit, and memory limit. Nothing else — no title, no explanations, no preamble.\n"
        "- The new problem has to be possible to pass with the constraints mentioned.\n"
        "- If the problem is so poorly written or ambiguous or non-algorithmic or that you cannot "
        "meaningfully buff/nerf it, output exactly: Sorry I can't do this task\n\n"
        f"Original problem:\n{problem}"
    )

    url     = f"{GEMINI_BASE}/models/{LLM_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1200},
    }

    try:
        resp = requests.post(url, json=payload, timeout=40)
        resp.raise_for_status()
        data   = resp.json()
        result = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return jsonify({"result": result})
    except requests.HTTPError:
        return jsonify({"error": _gemini_error(resp)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /api/solution ─────────────────────────────────────────────────────────────
@app.route("/api/solution", methods=["POST"])
def solution():
    """
    Generate a detailed solution for a competitive programming problem.

    Body:  { problem: str, techniques: str }
    Reply: { solution: str }
      - solution: full markdown solution with approach, algorithm, complexity analysis,
                  and annotated pseudocode / key implementation steps
    """
    body          = request.get_json(silent=True) or {}
    problem       = (body.get("problem")      or "").strip()
    techniques    = (body.get("techniques")   or "").strip()
    time_complex  = (body.get("timeComplex")  or "").strip()
    space_complex = (body.get("spaceComplex") or "").strip()
    special       = (body.get("special")      or "").strip()
    feasibility   = (body.get("feasibility")  or "practical").strip().lower()

    if not problem:
        return jsonify({"error": "Missing 'problem' field"}), 400
    if not GOOGLE_API_KEY:
        return jsonify({"error": "GOOGLE_API_KEY not configured on server"}), 500

    meta_lines = []
    meta_lines.append(f"- Feasibility: {feasibility} ({'fits within 1–2s and 512 MB with reasonable constraints' if feasibility == 'practical' else 'no reasonable constraint makes it fit within 1–2s; treat as a theoretical analysis'})")
    if techniques:                              meta_lines.append(f"- Techniques: {techniques}")
    if time_complex:                            meta_lines.append(f"- Required time complexity: {time_complex}")
    if space_complex:                           meta_lines.append(f"- Required space complexity: {space_complex}")
    if special and special.lower() != "none":   meta_lines.append(f"- Additional constraints: {special}")
    meta_block = ("Known problem metadata:\n" + "\n".join(meta_lines) + "\n\n") if meta_lines else ""

    prompt = (
        "You are a competitive programming coach writing a solution editorial.\n\n"
        f"{meta_block}"
        "Produce a concise but complete solution in Markdown with these sections:\n"
        "## Intuition\n"
        "1–3 sentences on the key observation that unlocks the problem.\n\n"
        "## Approach\n"
        "Step-by-step algorithm description. Wrap all math in $...$. "
        "Reference variable names from the problem statement. "
        "The approach MUST be consistent with the required time and space complexity above.\n\n"
        "## Complexity\n"
        "- **Time:** $O(...)$ — one-line justification.\n"
        "- **Space:** $O(...)$ — one-line justification.\n\n"
        "## Key Implementation Notes\n"
        "Bullet list of non-obvious implementation details, edge cases, or pitfalls. "
        "Include any relevant notes about the additional constraints (e.g. overflow, modular arithmetic).\n\n"
        "Rules:\n"
        "- Do NOT output a title or preamble — start directly with ## Intuition.\n"
        "- Do NOT include any code or pseudocode whatsoever.\n"
        "- Keep the total response under 450 words.\n"
        "- Use $ for all math expressions.\n\n"
        f"Problem:\n{problem}"
    )

    url     = f"{GEMINI_BASE}/models/{LLM_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1200},
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data     = resp.json()
        solution = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return jsonify({"solution": solution})
    except requests.HTTPError:
        return jsonify({"error": _gemini_error(resp)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/api/embed-tags", methods=["POST"])
def embed_tags():
    """
    Embed each tag individually using text-embedding-004, fanned out in parallel.

    Body:  { tags: ["bitmask DP", "0-1 BFS", ...] }
    Reply: { embeddings: { "bitmask DP": [0.1, ...], "0-1 BFS": [...], ... } }

    Partial success: if some tags fail, returns what succeeded plus a 'warnings' list.
    """
    body = request.get_json(silent=True) or {}
    tags = body.get("tags", [])

    if not tags or not isinstance(tags, list):
        return jsonify({"error": "Missing or invalid 'tags' field (expected list)"}), 400
    if not GOOGLE_API_KEY:
        return jsonify({"error": "GOOGLE_API_KEY not configured on server"}), 500

    # Deduplicate while preserving original casing
    unique_tags = list(dict.fromkeys(t.strip() for t in tags if t.strip()))

    embed_url = f"{GEMINI_BASE}/models/text-embedding-004:embedContent?key={GOOGLE_API_KEY}"

    def embed_one(tag: str):
        payload = {
            "model":   "models/text-embedding-004",
            "content": {"parts": [{"text": tag}]},
        }
        r = requests.post(embed_url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["embedding"]["values"]

    embeddings = {}
    errors     = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        future_to_tag = {pool.submit(embed_one, tag): tag for tag in unique_tags}
        for future in as_completed(future_to_tag):
            tag = future_to_tag[future]
            try:
                embeddings[tag] = future.result()
            except Exception as e:
                errors.append(f"{tag}: {e}")

    response = {"embeddings": embeddings}
    if errors:
        response["warnings"] = errors

    return jsonify(response)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _gemini_error(resp: requests.Response) -> str:
    try:
        return resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
    except Exception:
        return f"HTTP {resp.status_code}"


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 3667))
    print(f"[CP Graph] Serving on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)