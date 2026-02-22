import os
import time
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found. Please set it in your .env file.")

client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    """Render the main landing page."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Accept income & expense data and return AI-generated budget advice."""
    try:
        data = request.get_json(force=True)

        # --- Validate income ---------------------------------------------------
        income = data.get("income")
        if income is None or str(income).strip() == "":
            return jsonify({"error": "Monthly income is required."}), 400
        try:
            income = float(income)
            if income < 0:
                raise ValueError
        except (ValueError, TypeError):
            return jsonify({"error": "Income must be a valid positive number."}), 400

        # --- Validate expenses -------------------------------------------------
        expenses = data.get("expenses", {})
        if not isinstance(expenses, dict) or not expenses:
            return jsonify({"error": "At least one expense category is required."}), 400

        parsed_expenses = {}
        for category, amount in expenses.items():
            try:
                val = float(amount) if amount else 0.0
                if val < 0:
                    raise ValueError
                parsed_expenses[category] = val
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid amount for '{category}'."}), 400

        total_expenses = sum(parsed_expenses.values())

        # --- Build the prompt --------------------------------------------------
        expense_lines = "\n".join(
            f"  - {cat.replace('_', ' ').title()}: ₹{amt:,.2f}"
            for cat, amt in parsed_expenses.items()
        )

        prompt = f"""You are PocketSmart AI — a friendly, expert personal-finance advisor.

A user has shared the following monthly financial data:

Monthly Income : ₹{income:,.2f}
Total Expenses : ₹{total_expenses:,.2f}
Remaining      : ₹{income - total_expenses:,.2f}

Expense Breakdown:
{expense_lines}

Based on this data, provide a comprehensive yet easy-to-read budget analysis.
Structure your response with these sections using markdown headings (##):

## 📊 Budget Overview
Summarize income vs. expenses and the current financial health.

## 💡 Smart Saving Tips
Give 4-5 specific, actionable tips to reduce spending in the categories listed.

## 📈 Better Spending Strategies
Suggest how to reallocate money for better value (e.g., invest, emergency fund).

## 🎯 Recommended Monthly Budget
Provide an ideal budget split (table format) for someone with this income level.

## ⚠️ Warnings
Flag any concerning patterns (overspending, no savings, etc.).

Keep the tone encouraging and practical. Use bullet points and emojis for readability.
"""

        # --- Call Groq with retry for rate limits ------------------------------
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are PocketSmart AI, an expert personal-finance advisor. Respond with well-structured markdown.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=2048,
                )
                advice = response.choices[0].message.content
                return jsonify({"advice": advice})
            except Exception as api_err:
                err_str = str(api_err).lower()
                if "rate" in err_str or "429" in err_str or "limit" in err_str:
                    if attempt < max_retries - 1:
                        app.logger.warning(f"Rate limited (attempt {attempt+1}), retrying in 10s...")
                        time.sleep(10)
                        continue
                    else:
                        app.logger.error(f"Rate limit exceeded after {max_retries} retries: {api_err}")
                        return jsonify({
                            "error": "Groq API rate limit reached. Please wait 30-60 seconds and try again."
                        }), 429
                else:
                    raise api_err

    except Exception as exc:
        app.logger.error(f"Analysis error: {exc}")
        error_msg = str(exc).lower()
        if "api_key" in error_msg or "invalid" in error_msg or "authenticate" in error_msg:
            return jsonify({"error": "Invalid API key. Please check your GROQ_API_KEY in the .env file."}), 401
        return jsonify({"error": "Something went wrong while generating advice. Please try again."}), 500


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)
