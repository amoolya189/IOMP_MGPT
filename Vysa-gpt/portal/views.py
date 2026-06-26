from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from .utils import gemini_answer


# ---------------- PAGES ----------------

def index(request):
    return render(request, "portal/index.html")


def chatbot(request):
    return render(request, "portal/chatbot.html")


def family_tree(request):
    return render(request, "portal/family_tree.html")

def game(request):
    return render(request, 'portal/game.html')

def quiz(request):
    return render(request, 'portal/quiz.html')


# ---------------- CHATBOT API ----------------

@csrf_exempt
def api_ask(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
        question = body.get("question", "").strip()
    except Exception:
        question = ""

    if not question:
        return JsonResponse({"answer": "<p>Please ask a Mahabharata-related question.</p>"})

    # Retrieve previous valid Mahabharata answer
    previous_answer = request.session.get("previous_answer", "")

    answer = gemini_answer(question, previous_answer)

    # ❗ Detect rejection response
    rejection_phrases = [
        "outside the scope",
        "not related to the mahabharata",
        "mahabharata-only",
        "within the world of the epic",
        "focus on the mahabharata"
    ]

    # Save context ONLY if answer is valid
    if not any(phrase in answer.lower() for phrase in rejection_phrases):
        request.session["previous_answer"] = answer

    return JsonResponse({"answer": answer})
