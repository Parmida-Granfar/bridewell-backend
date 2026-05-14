"""Import normalized student log JSON datasets into ChatMessage."""

from pathlib import Path

from django.core.management.base import BaseCommand

from bridewell_api.models import ChatMessage
from bridewell_api.studentlog_utils import parse_studentlog_dataset


class Command(BaseCommand):
    help = "Import student log datasets into the ChatMessage table."

    def add_arguments(self, parser):
        parser.add_argument("path", type=str, help="Path to the student log JSON file or directory")
        parser.add_argument(
            "--clear-source",
            action="store_true",
            help="Delete existing ChatMessage rows imported from studentlogs before importing.",
        )

    def handle(self, *args, **options):
        path = Path(options["path"])
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if options["clear_source"]:
            deleted, _ = ChatMessage.objects.filter(source="studentlogs").delete()
            self.stdout.write(self.style.WARNING(f"Deleted {deleted} existing studentlog messages"))

        messages = parse_studentlog_dataset(path)
        if not messages:
            self.stdout.write(self.style.WARNING("No student log messages found in input."))
            return

        objs = [ChatMessage(**msg) for msg in messages]
        ChatMessage.objects.bulk_create(objs, batch_size=500)
        self.stdout.write(self.style.SUCCESS(f"Imported {len(objs)} student log messages from {path}"))
