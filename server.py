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

    prompt = (
        "You are a competitive programming expert. "
        "Analyze the problem and output EXACTLY 5 lines, no labels, no preamble.\n\n"
        "Line 1: comma-separated specific CP techniques (e.g. 'Bitmask DP, Binary Search on Answer', not bare 'DP' or 'Graph'). "
        "Never include trivial implementation details: no recursion, iteration, loops, arrays, sorting (unless it is the key insight), input parsing, brute force, or simulation. "
        "Each technique must use its single canonical name — no alternatives or parenthetical aliases (e.g. 'DSU', not 'Union-Find (DSU)' or 'Union-Find'; 'Fenwick Tree', not 'BIT' or 'Binary Indexed Tree'). "
        "Capitalize each technique in title case (e.g. 'Segment Tree with Lazy Propagation', 'Binary Search on Answer', 'DSU on Tree').\n"
        "Line 2: tightest Big-O time complexity an accepted solution must meet (e.g. O(N log N)).\n"
        "Line 3: tightest Big-O space complexity (e.g. O(N)).\n"
        "Line 4: any non-obvious special constraints (online queries, overflow risk, etc.) — or exactly 'None'.\n"
        "Line 5: one sentence — what is given, what operations exist, what the goal is. "
        "Example: 'Given an array of N integers, find the subarray with the largest sum.'\n\n"
        f"Problem:\n{problem}"
    )

    url     = f"{GEMINI_BASE}/models/{LLM_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 200},
    }

    try:
        resp = requests.post(url, json=payload, timeout=40)
        resp.raise_for_status()
        data = resp.json()
        raw   = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Parse the 5-line structured output
        # Line 1: techniques, Line 2: time complexity, Line 3: space complexity
        # Line 4: special constraints, Line 5+: summary (may be multi-line)
        lines = raw.split("\n")
        techniques   = lines[0].strip() if len(lines) > 0 else ""
        time_complex = lines[1].strip() if len(lines) > 1 else ""
        space_complex = lines[2].strip() if len(lines) > 2 else ""
        special      = lines[3].strip() if len(lines) > 3 else "None"
        summary      = "\n".join(lines[4:]).strip() if len(lines) > 4 else ""

        return jsonify({
            "techniques":    techniques,
            "timeComplex":   time_complex,
            "spaceComplex":  space_complex,
            "special":       special,
            "summary":       summary,
        })
    except requests.HTTPError:
        return jsonify({"error": _gemini_error(resp)}), 502
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
    problem = (body.get("problem") or "").strip()
    mode    = (body.get("mode")    or "").strip().lower()

    if not problem:
        return jsonify({"error": "Missing 'problem' field"}), 400
    if mode not in ("buff", "nerf"):
        return jsonify({"error": "Field 'mode' must be 'buff' or 'nerf'"}), 400
    if not GOOGLE_API_KEY:
        return jsonify({"error": "GOOGLE_API_KEY not configured on server"}), 500

    if mode == "buff":
        direction_instruction = (
            "BUFF the problem: make it strictly HARDER. "
            "The problem must require a more optimal algorithm. "
            "The core algorithmic idea should stay relatively similar."
        )
    else:
        direction_instruction = (
            "NERF the problem: make it strictly EASIER. "
            "The problem should allow a simpler algorithm to pass. "
            "The core algorithmic idea should stay relatively similar."
        )

    prompt = (
        "You are a competitive programming problem setter.\n\n"
        f"{direction_instruction}\n\n"
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


# ── /api/embed-tags ───────────────────────────────────────────────────────────
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