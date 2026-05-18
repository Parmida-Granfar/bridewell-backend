from django.urls import path

from .views import (
    BehaviorMixView,
    ClassSummaryView,
    CognitiveLoadView,
    EngagementTimelineView,
    InteractionSummaryView,
    TopicWrestlingView,
    chat_summary,
    suggested_pair_ups,
    learning_preferences_summary,
)

app_name = "bridewell_api"

urlpatterns = [
    path("cognitive-load/", CognitiveLoadView.as_view(), name="cognitive-load"),
    path("topic-wrestling/", TopicWrestlingView.as_view(), name="topic-wrestling"),
    path("class-summary/", ClassSummaryView.as_view(), name="class-summary"),
    path("engagement-timeline/", EngagementTimelineView.as_view(), name="engagement-timeline"),
    path("behavior-mix/", BehaviorMixView.as_view(), name="behavior-mix"),
    path("interaction-summary/", InteractionSummaryView.as_view(), name="interaction-summary"),
    path("chat-summary/<str:student_id>/", chat_summary, name="chat-summary"),
    path("pair-ups/", suggested_pair_ups, name="pair-ups"),
    path("learning-preferences/<str:student_id>/", learning_preferences_summary, name="learning-preferences"),
]