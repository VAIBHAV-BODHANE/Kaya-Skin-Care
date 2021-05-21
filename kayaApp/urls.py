from django.urls import path, include
from kayaApp import views

from rest_framework.authtoken.views import obtain_auth_token

urlpatterns = [
    # path('home', views.home, name="home"),
    # path('signup', views.signup, name='sign-up'),
    path('registeruser', views.RegisterUserView.as_view(), name='register_user'),
    path('doctor_info/', views.UserInfoView.as_view(), name='doctor_info'),
    # path('login', views.login, name='login'),
    path('login', views.LoginView.as_view(), name='login'),
    path('logout', views.LogoutView.as_view(), name='logout'),
    path('password_reset/', include('django_rest_passwordreset.urls', namespace='password_reset')),
    path('password_reset/confirm/', include('django_rest_passwordreset.urls', namespace='password_reset_confirm')),
    path('patient/', views.PatientView.as_view(), name='patient-details'),
    path('patient/<int:pk>', views.PatientUpdateDelete.as_view(), name='patient-update-delete'),
    # path('patient_image/', views.PatientImageList.as_view(), name='patient-image'),
    path('patient_image/<int:pk>', views.PatientImageList.as_view(), name='patient-image'),
    path('form1_all_questions/', views.QuestionAnswer1.as_view(), name='question-answer-form1'),
    path('form2_all_questions/<int:pk>', views.QuestionAnswer2.as_view(), name='question-answer-form2'),
    path('patient_form1/<int:pk>', views.PatientForm1List.as_view(), name='patient-form1'),
    path('patient_form1/<int:fk>/<int:pk>', views.PatientForm1List.as_view(), name='patient-form1'),
    path('patient_form2/<int:pk>', views.PatientForm2List.as_view(), name='patient-form2'),
    path('patient_form2/<int:fk1>/<int:pk>', views.PatientForm2List.as_view(), name='patient-form2'),
    path('patient_form2/<int:fk1>/<int:fk2>/<int:pk>', views.PatientForm2List.as_view(), name='patient-form2'),
    path('diagnose_image/<int:pk>', views.PatientDiagnoseImage.as_view(), name='diagnose-image'),
    # path('patientbeforeafter/', views.PatientBeforeAfterImageView.as_view(), name='patientbeforeafter'),
    path('patient_before_after/<int:pk>', views.PatientBeforeAfterImageView.as_view(), name='patientbeforeafter'),
    # path('patientmajorimprovement/', views.PatientMajorImprovementView.as_view(), name='patientmajorimprovement'),
    path('patient_measure_improvement/<int:pk>', views.PatientMeasureImprovementView.as_view(), name='patientmajorimprovement'),
    path('servicelist/<int:pk>', views.ServiceView.as_view(), name='servicelist'),
    # path('productlist/<int:pk>', views.ProductView.as_view(), name='productlist'),
    path('check_form1/<int:pk>', views.CheckForm1.as_view(), name='check-form1'),
    path('check_form2/<int:pk>', views.CheckForm2.as_view(), name='check-form2'),
    path('check_service_product/<int:pk>', views.CheckData.as_view(), name='check-service-product'),
    path('patient_name_phone_info/', views.PatientNamePhoneInfoView.as_view(), name='patient_name_phone_info'),
    # path('testapi/', views.testing_from, name='testing_from'),



]