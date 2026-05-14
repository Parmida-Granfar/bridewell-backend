from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bridewell_api', '0002_studentpassport'),
    ]

    operations = [
        migrations.AddField(
            model_name='chatmessage',
            name='source',
            field=models.CharField(blank=True, db_index=True, default='', help_text="Optional origin of the message, e.g. 'studentlogs'.", max_length=64),
        ),
        migrations.AddField(
            model_name='chatmessage',
            name='source_id',
            field=models.CharField(blank=True, default='', help_text='Optional original message ID from the source log.', max_length=128),
        ),
    ]
