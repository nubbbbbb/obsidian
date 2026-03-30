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


# ── /api/extract ──────────────────────────────────────────────────────────────
@app.route("/api/extract", methods=["POST"])
def extract():
    """
    Extract algorithmic techniques from a competitive-programming problem.
    Uses the model specified by LLM_MODEL in .env.
    Returns { techniques: str }  (comma-separated specific technique names).
    """
    body    = request.get_json(silent=True) or {}
    problem = (body.get("problem") or "").strip()

    if not problem:
        return jsonify({"error": "Missing 'problem' field"}), 400
    if not GOOGLE_API_KEY:
        return jsonify({"error": "GOOGLE_API_KEY not configured on server"}), 500

    prompt = (
        "You are a competitive programming expert. "
        "Analyze the following problem and identify the specific algorithmic techniques required to solve it.\n\n"
        "Rules:\n"
        "- Only include REAL competitive programming techniques that are non-trivial and problem-specific.\n"
        "- ALWAYS use the most specific variant of a technique - never output a bare category name. "
        "Qualify every technique to its exact subtype. Examples of required specificity:\n"
        "  * DP must be qualified: 2D DP, bitmask DP, digit DP, interval DP, knapsack DP, "
        "DP on trees, DP on DAG, DP with monotone deque, broken profile DP, "
        "DP with divide and conquer optimization, DP with convex hull trick.\n"
        "  * Graph traversal must be qualified: multi-source BFS, 0-1 BFS, BFS on grid, "
        "Dijkstra on implicit graph, Bellman-Ford, Floyd-Warshall, SCC (Tarjan), "
        "SCC (Kosaraju), bipartite matching, topological sort, Euler path/circuit.\n"
        "  * Segment tree must be qualified: segment tree with lazy propagation, "
        "persistent segment tree, segment tree beats, merge sort tree, 2D segment tree.\n"
        "  * Binary search must be qualified: binary search on answer, "
        "binary search on floating point, parallel binary search.\n"
        "  * Greedy must be qualified: greedy with sorting, greedy with priority queue, "
        "exchange argument greedy.\n"
        "  * String algorithms must be specific: KMP, Z-algorithm, Aho-Corasick, "
        "suffix array, suffix automaton, rolling hash.\n"
        "  * Tree techniques must be specific: LCA with binary lifting, LCA with Euler tour, "
        "HLD (heavy-light decomposition), centroid decomposition, "
        "small-to-large merging, DSU on tree.\n"
        "  * Math must be specific: matrix exponentiation, inclusion-exclusion, "
        "Mobius inversion, Euler totient, Lucas theorem, NTT, FFT, Gaussian elimination.\n"
        "  * Other fine-grained examples: two pointers, sliding window, prefix sum, "
        "difference array, sparse table (RMQ), union-find (DSU), "
        "binary indexed tree (Fenwick tree), Mo's algorithm, "
        "sqrt decomposition, offline processing, coordinate compression, hashing.\n"
        "- Do NOT output bare category names like 'dynamic programming', 'graph', "
        "'segment tree', 'binary search', or 'greedy' alone - always add the subtype.\n"
        "- Do NOT include generic constructs: no 'recursion', 'iteration', 'loops', "
        "'arrays', 'sorting' (unless sorting IS the key insight), "
        "'input parsing', 'brute force', or 'simulation' "
        "(unless simulation is the deliberate intended solution).\n"
        "- Omit sub-steps that are not independently noteworthy "
        "(e.g. do not list 'sorting' if it merely precedes a binary search on answer).\n\n"
        "Output ONLY a comma-separated list of specific technique names, nothing else.\n\n"
        f"Problem:\n{problem}"
    )

    url     = f"{GEMINI_BASE}/models/{LLM_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 200},
    }

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data       = resp.json()
        techniques = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return jsonify({"techniques": techniques})
    except requests.HTTPError:
        return jsonify({"error": _gemini_error(resp)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /api/summarize ────────────────────────────────────────────────────────────
@app.route("/api/summarize", methods=["POST"])
def summarize():
    """
    Summarize a competitive-programming problem into compact markdown.

    Body:  { problem: str }
    Reply: { summary: str }
      - One-line tags header (bold, comma-separated key constraints/numbers)
        followed by a short markdown summary covering problem type, constraints,
        and what makes it non-trivial. No preamble, no title.
    """
    body    = request.get_json(silent=True) or {}
    problem = (body.get("problem") or "").strip()

    if not problem:
        return jsonify({"error": "Missing 'problem' field"}), 400
    if not GOOGLE_API_KEY:
        return jsonify({"error": "GOOGLE_API_KEY not configured on server"}), 500

    prompt = (
        "You are a competitive programming assistant. "
        "Given the problem below, produce a concise markdown summary.\n\n"
        "Format — output EXACTLY these two sections, nothing else:\n\n"
        "Line 1: A single line of the key numeric constraints and input type, "
        "formatted as bold inline tags separated by commas. "
        "Example: `**N ≤ 2×10⁵**, **Q ≤ 10⁵**, **weighted tree**, **1 ≤ w ≤ 10⁹**`\n\n"
        "Then a blank line.\n\n"
        "Then a short markdown paragraph (3–6 sentences) summarising: "
        "what the problem asks, the key structural insight or difficulty, "
        "and the binding constraints. "
        "Use inline code for variable names (e.g. `N`, `K`). "
        "Do NOT mention algorithm names or solution approaches.\n\n"
        f"Problem:\n{problem}"
    )

    url     = f"{GEMINI_BASE}/models/{LLM_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 350},
    }

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data    = resp.json()
        summary = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return jsonify({"summary": summary})
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