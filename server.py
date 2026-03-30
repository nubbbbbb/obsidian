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
        "Analyze the problem below and produce TWO sections separated by exactly '---'.\n\n"

        "SECTION 1 — Techniques:\n"
        "Output a single comma-separated list of the specific algorithmic techniques "
        "required to solve this problem. Rules:\n"
        "- Only REAL, non-trivial, problem-specific CP techniques.\n"
        "- Always use the most specific subtype — never a bare category name:\n"
        "  DP subtypes: 2D DP, bitmask DP, digit DP, interval DP, knapsack DP, "
        "DP on trees, DP on DAG, DP with monotone deque, broken profile DP, "
        "DP with divide and conquer optimization, DP with convex hull trick.\n"
        "  Graph subtypes: multi-source BFS, 0-1 BFS, BFS on grid, "
        "Dijkstra on implicit graph, Bellman-Ford, Floyd-Warshall, SCC (Tarjan), "
        "SCC (Kosaraju), bipartite matching, topological sort, Euler path/circuit.\n"
        "  Segment tree subtypes: segment tree with lazy propagation, "
        "persistent segment tree, segment tree beats, merge sort tree, 2D segment tree.\n"
        "  Binary search subtypes: binary search on answer, "
        "binary search on floating point, parallel binary search.\n"
        "  Greedy subtypes: greedy with sorting, greedy with priority queue, "
        "exchange argument greedy.\n"
        "  Strings: KMP, Z-algorithm, Aho-Corasick, suffix array, suffix automaton, rolling hash.\n"
        "  Trees: LCA with binary lifting, LCA with Euler tour, HLD, centroid decomposition, "
        "small-to-large merging, DSU on tree.\n"
        "  Math: matrix exponentiation, inclusion-exclusion, Mobius inversion, "
        "Euler totient, Lucas theorem, NTT, FFT, Gaussian elimination.\n"
        "  Other: two pointers, sliding window, prefix sum, difference array, "
        "sparse table (RMQ), union-find (DSU), binary indexed tree (Fenwick tree), "
        "Mo's algorithm, sqrt decomposition, offline processing, "
        "coordinate compression, hashing.\n"
        "- Do NOT output bare names like 'dynamic programming', 'graph', "
        "'segment tree', 'binary search', or 'greedy' alone.\n"
        "- Do NOT include: recursion, iteration, loops, arrays, sorting (unless it IS "
        "the key insight), input parsing, brute force, simulation (unless deliberate).\n"
        "- Omit sub-steps that are not independently noteworthy.\n\n"

        "SECTION 2 — Summary:\n"
        "A concise markdown summary in EXACTLY this format:\n"
        "  Line 1: key numeric constraints and input type as bold inline tags, "
        "comma-separated. Example: **N ≤ 2×10⁵**, **Q ≤ 10⁵**, **weighted tree**, **1 ≤ w ≤ 10⁹**\n"
        "  Blank line.\n"
        "  One markdown paragraph (3–6 sentences): what the problem asks, "
        "the key structural difficulty, and the binding constraints. "
        "Use inline code for variable names. Do NOT name algorithms or solution approaches.\n\n"

        "Output format — EXACTLY:\n"
        "<comma-separated techniques>\n"
        "---\n"
        "<summary>\n\n"
        f"Problem:\n{problem}"
    )

    url     = f"{GEMINI_BASE}/models/{LLM_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 600},
    }

    try:
        resp = requests.post(url, json=payload, timeout=40)
        resp.raise_for_status()
        data = resp.json()
        raw  = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Split on the first '---' separator
        parts      = raw.split("---", 1)
        techniques = parts[0].strip()
        summary    = parts[1].strip() if len(parts) > 1 else ""

        return jsonify({"techniques": techniques, "summary": summary})
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