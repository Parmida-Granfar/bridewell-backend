"""Import a student passport DOCX into the StudentPassport model."""

from pathlib import Path

from django.core.management.base import BaseCommand

from bridewell_api.models import StudentPassport
from bridewell_api.passport_utils import parse_passport_docx


class Command(BaseCommand):
    help = "Import a student passport DOCX and persist it to the database."

    def add_arguments(self, parser):
        parser.add_argument("path", type=str, help="Path to the passport .docx file")
        parser.add_argument(
            "--student-id",
            type=str,
            help="Optional student_id if the document does not include a clear identifier",
        )

    def handle(self, *args, **options):
        path = Path(options["path"])
        if not path.exists():
            raise FileNotFoundError(f"Passport file not found: {path}")

        passport_data = parse_passport_docx(path)
        student_id = passport_data.get("student_id") or options.get("student_id")
        if not student_id:
            raise ValueError(
                "Student identifier could not be extracted from the passport. "
                "Provide --student-id or include a visible student name/ID in the document."
            )

        passport_data["student_id"] = student_id
        passport_data["source_file"] = str(path)

        passport, created = StudentPassport.objects.update_or_create(
            student_id=student_id,
            defaults={
                "access_arrangements": passport_data.get("access_arrangements", []),
                "declared_needs": passport_data.get("declared_needs", []),
                "preferred_mode": passport_data.get("preferred_mode", ""),
                "support_needs": passport_data.get("support_needs", []),
                "raw_text": passport_data.get("raw_text", ""),
                "source_file": passport_data.get("source_file", ""),
            },
        )

        action = "Created" if created else "Updated"
        self.stdout.write(self.style.SUCCESS(f"{action} passport for student_id={student_id}"))
