from django.db import models
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.urls import reverse
from django_rest_passwordreset.signals import reset_password_token_created
from django.core.mail import send_mail

from rest_framework.authtoken.models import Token


class MyUserManager(BaseUserManager):
    def create_user(self, email, username, fname, lname, password=None):
        if not email:
            raise ValueError('user must have an email address!')
        if not username:
            raise ValueError('user must have an username')

        user = self.model(
            email = self.normalize_email(email),
            username = self.username,
            fname = fname,
            lname = lname,
        )

        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, username, fname, lname, password):
        user = self.create_user(
            email = self.normalize_email(email),
            username = username,
            fname = fname,
            lname = lname,
            password = password,
        )
        user.is_admin = True
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)
        return user


class UserRegister(AbstractBaseUser):
    id = models.AutoField(primary_key=True)
    email = models.CharField(max_length=150, unique=True)
    username = models.CharField(max_length=50, unique=True)
    fname = models.CharField(max_length=50)
    lname = models.CharField(max_length=50)
    password = models.CharField(max_length=100)
    
    USERNAME_FIELD = 'email'
    # REQUIRED_FIELDS = ['username']

    objects = MyUserManager()

    def __str__(self):
        return self.email
        

@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.create(user=instance)


class PatientDetails(models.Model):
    doctor = models.ForeignKey(UserRegister, related_name='doctor', on_delete=models.CASCADE)
    patient_name = models.CharField(max_length=100)
    patient_email = models.CharField(max_length=255)
    age = models.IntegerField()
    gender = models.CharField(max_length=20)
    dob = models.DateField(null=True)
    phone = models.CharField(max_length=20)
    regi_date = models.DateField(null=True)
    heartbeat = models.IntegerField(null=True)
    bp = models.CharField(max_length=10, null=True)
    dust_allergy = models.CharField(max_length=50, null=True)
    medication = models.CharField(max_length=100, null=True)

    def __str__(self):
        return self.patient_name


@receiver(reset_password_token_created)
def password_reset_token_created(sender, instance, reset_password_token, *args, **kwargs):

    email_plaintext_message = "token={}".format(reset_password_token.key)

    send_mail(
        # title:
        "Password Reset for {title}".format(title="Kaya App"),
        # message:
        email_plaintext_message,
        # from:
        "noreply@somehost.local",
        # to:
        [reset_password_token.user.email]
    )


class PatientImage(models.Model):
    patient = models.ForeignKey(PatientDetails, related_name='patient', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='kayaApp/patient/images')
    # remark = models
    # position = (
    #     ('L','Left'),
    #     ('R','Right'),
    #     ('F','Front'),
    # )
    # image_position = models.CharField(choices=position, max_length=6, default='F')
    created_date = models.DateField(auto_now_add=True)
    updated_date = models.DateField(auto_now=True)

    def __str__(self):
        return str(self.patient)



class QuestionAnswersForm1(models.Model):
    question_type = models.JSONField()

    def __str__(self):
        return self.question_type['question']


class QuestionAnswersForm2(models.Model):
    question_type = models.JSONField()

    def __str__(self):
        return self.question_type['question']


class PatientForm1(models.Model):
    patient = models.ForeignKey(PatientDetails, on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField()
    answer_type = models.TextField(null=True)
    created_date = models.DateField(auto_now_add=True)
    updated_date = models.DateField(auto_now=True)

    def __str__(self):
        return self.question


class PatientForm2(models.Model):
    patient = models.ForeignKey(PatientDetails, on_delete=models.CASCADE)
    form2_concern1 = models.ForeignKey(QuestionAnswersForm2, related_name='concern1', on_delete=models.CASCADE, null=True)
    form2_concern2 = models.ForeignKey(QuestionAnswersForm2, related_name='concern2', on_delete=models.CASCADE, null=True)
    question = models.TextField(max_length=1000)
    answer = models.TextField(max_length=1000)
    skin_type = models.TextField()
    answer_type = models.TextField()
    created_date = models.DateField(auto_now_add=True)
    updated_date = models.DateField(auto_now=True)

    def __str__(self):
        return self.question


# class Product(models.Model):
#     product_name = models.TextField(max_length=500)
#     product_image = models.ImageField(upload_to='kayaApp/product/images')

#     def __str__(self):
#         return self.product_name


class DiagnoseImage(models.Model):
    patient = models.ForeignKey(PatientDetails, related_name="patient_id", on_delete = models.CASCADE)
    d_image = models.ImageField(upload_to='kayaApp/diagnose_image/images')
    process_image = models.ImageField(upload_to='kayaApp/diagnose_process_image/images')
    created_date = models.DateField(auto_now_add=True)
    updated_date = models.DateField(auto_now=True)


class PatientBeforeAfter(models.Model):
    patient = models.ForeignKey(PatientDetails,related_name='Patient_image',on_delete=models.CASCADE)
    image = models.ImageField(upload_to='kayaApp/patient_before_after/images')
    process_image = models.ImageField(upload_to='kayaApp/patient_before_after_process/images', null=True)
    created_date = models.DateField(auto_now_add=True)
    updated_date = models.DateField(auto_now=True)

    def __str__(self):
        return self.patient

class PatientMeasureImprovement(models.Model):
    patient = models.ForeignKey(PatientDetails,related_name='Patient_improvement',on_delete=models.CASCADE)
    remark = models.BooleanField(null = True)
    image = models.ImageField(upload_to='kayaApp/patient_measure_improvement/images')
    process_image = models.ImageField(upload_to='kayaApp/patient_measure_improvement_process/images',null = True)
    skin_score = models.FloatField(null=True)
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return str(self.patient)


class ConcernList(models.Model):
    concern = models.JSONField()

    def __str__(self):
        return str(self.concern)



class ServiceList(models.Model):
    service = models.JSONField()
    concern = models.ForeignKey(ConcernList, on_delete=models.CASCADE)

    def __str__(self):
        return str(self.service)


class ProductList(models.Model):
    product = models.JSONField()
    service = models.ForeignKey(ServiceList, on_delete=models.CASCADE)

    def __str__(self):
        return str(self.product)


class PatientServiceProduct(models.Model):
    service = models.TextField()
    product = models.TextField()
    patient = models.ForeignKey(PatientDetails, on_delete=models.CASCADE)

    def __str__(self):
        return str(self.service)