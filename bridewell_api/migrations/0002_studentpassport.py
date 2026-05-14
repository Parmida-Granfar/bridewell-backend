from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bridewell_api', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='StudentPassport',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('student_id', models.CharField(db_index=True, help_text='Student identifier used to join passport data with chat logs.', max_length=64, unique=True)),
                ('access_arrangements', models.JSONField(blank=True, default=list)),
                ('declared_needs', models.JSONField(blank=True, default=list)),
                ('preferred_mode', models.CharField(blank=True, max_length=128)),
                ('support_needs', models.JSONField(blank=True, default=list)),
                ('raw_text', models.TextField(blank=True)),
                ('source_file', models.CharField(blank=True, max_length=256)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'indexes': [models.Index(fields=['student_id'], name='bridewell_a_student_407e52_idx')],
            },
        ),
    ]
