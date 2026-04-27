from flask import Flask, request, jsonify, render_template
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# ✅ Safe API key load
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment")

client = Groq(api_key=api_key)

# ── System Prompt ──
SYSTEM_PROMPT = """You are MediBot, a helpful health assistant...
(keep your same prompt here, no change needed)
"""

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        # ✅ SAFE JSON handling (IMPORTANT FIX)
        data = request.get_json(silent=True) or {}

        user_message = data.get("message", "").strip()
        history = data.get("history", [])

        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for h in history[-10:]:
            messages.append({
                "role": h.get("role", "user"),
                "content": h.get("content", "")
            })

        messages.append({"role": "user", "content": user_message})

        # AI call
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=512,
            temperature=0.6,
        )

        reply = response.choices[0].message.content.strip()

        return jsonify({
            "reply": reply,
            "safe_flag": False
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)