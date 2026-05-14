"""
App configuration for bridewell_api.
"""

from django.apps import AppConfig


class BridewellApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'bridewell_api'
    verbose_name = 'Bridewell AI Metrics API'