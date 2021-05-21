# Generated by Django 3.1.5 on 2021-03-13 14:29

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='UserRegister',
            fields=[
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='last login')),
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('email', models.CharField(max_length=150, unique=True)),
                ('username', models.CharField(max_length=50, unique=True)),
                ('fname', models.CharField(max_length=50)),
                ('lname', models.CharField(max_length=50)),
                ('password', models.CharField(max_length=100)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='ConcernList',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('concern', models.JSONField()),
            ],
        ),
        migrations.CreateModel(
            name='PatientDetails',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('patient_name', models.CharField(max_length=100)),
                ('patient_email', models.CharField(max_length=255)),
                ('age', models.IntegerField()),
                ('gender', models.CharField(max_length=20)),
                ('dob', models.DateField(null=True)),
                ('phone', models.CharField(max_length=20)),
                ('regi_date', models.DateField(null=True)),
                ('heartbeat', models.IntegerField(null=True)),
                ('bp', models.CharField(max_length=10, null=True)),
                ('dust_allergy', models.CharField(max_length=50, null=True)),
                ('medication', models.CharField(max_length=100, null=True)),
                ('doctor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='doctor', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='QuestionAnswersForm1',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question_type', models.JSONField()),
            ],
        ),
        migrations.CreateModel(
            name='QuestionAnswersForm2',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question_type', models.JSONField()),
            ],
        ),
        migrations.CreateModel(
            name='ServiceList',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('service', models.JSONField()),
                ('concern', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='kayaApp.concernlist')),
            ],
        ),
        migrations.CreateModel(
            name='ProductList',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('product', models.JSONField()),
                ('service', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='kayaApp.servicelist')),
            ],
        ),
        migrations.CreateModel(
            name='PatientServiceProduct',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('service', models.TextField()),
                ('product', models.TextField()),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='kayaApp.patientdetails')),
            ],
        ),
        migrations.CreateModel(
            name='PatientMeasureImprovement',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('remark', models.BooleanField(null=True)),
                ('image', models.ImageField(upload_to='kayaApp/patient_measure_improvement/images')),
                ('process_image', models.ImageField(null=True, upload_to='kayaApp/patient_measure_improvement_process/images')),
                ('skin_score', models.FloatField(null=True)),
                ('created_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(auto_now=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='Patient_improvement', to='kayaApp.patientdetails')),
            ],
        ),
        migrations.CreateModel(
            name='PatientImage',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='kayaApp/patient/images')),
                ('created_date', models.DateField(auto_now_add=True)),
                ('updated_date', models.DateField(auto_now=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='patient', to='kayaApp.patientdetails')),
            ],
        ),
        migrations.CreateModel(
            name='PatientForm2',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.TextField(max_length=1000)),
                ('answer', models.TextField(max_length=1000)),
                ('skin_type', models.TextField()),
                ('answer_type', models.TextField()),
                ('created_date', models.DateField(auto_now_add=True)),
                ('updated_date', models.DateField(auto_now=True)),
                ('form2_concern1', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='concern1', to='kayaApp.questionanswersform2')),
                ('form2_concern2', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='concern2', to='kayaApp.questionanswersform2')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='kayaApp.patientdetails')),
            ],
        ),
        migrations.CreateModel(
            name='PatientForm1',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.TextField()),
                ('answer', models.TextField()),
                ('answer_type', models.TextField(null=True)),
                ('created_date', models.DateField(auto_now_add=True)),
                ('updated_date', models.DateField(auto_now=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='kayaApp.patientdetails')),
            ],
        ),
        migrations.CreateModel(
            name='PatientBeforeAfter',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='kayaApp/patient_before_after/images')),
                ('process_image', models.ImageField(null=True, upload_to='kayaApp/patient_before_after_process/images')),
                ('created_date', models.DateField(auto_now_add=True)),
                ('updated_date', models.DateField(auto_now=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='Patient_image', to='kayaApp.patientdetails')),
            ],
        ),
        migrations.CreateModel(
            name='DiagnoseImage',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('d_image', models.ImageField(upload_to='kayaApp/diagnose_image/images')),
                ('process_image', models.ImageField(upload_to='kayaApp/diagnose_process_image/images')),
                ('created_date', models.DateField(auto_now_add=True)),
                ('updated_date', models.DateField(auto_now=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='patient_id', to='kayaApp.patientdetails')),
            ],
        ),
    ]
