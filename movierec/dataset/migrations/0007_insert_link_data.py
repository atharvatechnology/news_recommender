# Generated by Django 3.2.13 on 2023-02-09 09:37

from django.db import migrations
def import_link_data(myapp, schema_editor):
    from django.core.management import call_command
    call_command('import_link_data')

class Migration(migrations.Migration):

    dependencies = [
        ('dataset', '0006_link'),
    ]

    operations = [
        migrations.RunPython(import_link_data),
    ]
