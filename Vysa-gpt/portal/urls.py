from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("chatbot/", views.chatbot, name="chatbot"),
    path("family-tree/", views.family_tree, name="family_tree"),
    path("api/ask/", views.api_ask, name="api_ask"),
    path("game/", views.game, name='game'),
    path("quiz/", views.quiz, name='quiz'),

]
