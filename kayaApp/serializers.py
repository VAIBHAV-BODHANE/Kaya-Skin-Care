from rest_framework import serializers

from kayaApp.models import UserRegister, PatientDetails, PatientImage, PatientForm1, PatientForm2, QuestionAnswersForm1, QuestionAnswersForm2, DiagnoseImage,PatientBeforeAfter,PatientMeasureImprovement, PatientServiceProduct


class UserRegisterSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = UserRegister
        fields = ('id', 'email', 'username', 'fname', 'lname', 'password')
        extra_kwargs = {
            'password': {'write_only': True}
        }

    def save(self):
        data = UserRegister(
            email=self.validated_data['email'],
            username=self.validated_data['username'],
            fname=self.validated_data['fname'],
            lname=self.validated_data['lname'],
        )
        password=self.validated_data['password']
        data.set_password(password)
        data.save()
        return data
    

class LoginSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(required=True)
    password = serializers.CharField(
        style={'input_type': 'password'}, write_only=True, required=True
    )

    class Meta:
        model = UserRegister
        fields = (
            'email',
            'password'
        )

class PatientDetailsSerializer(serializers.ModelSerializer):

    doctor = serializers.RelatedField(source='UserRegister', read_only=True)
    
    class Meta:
        model = PatientDetails
        fields = '__all__'

    # def save(self):
    #     data = PatientDetails(
    #         doctor=self.validated_data['doctor'],
    #         patient_name=self.validated_data['patient_name'],
    #         age=self.validated_data['age'],
    #         gender=self.validated_data['gender'],
    #         dob=self.validated_data['dob'],
    #         phone=self.validated_data['phone'],
    #         regi_date=self.validated_data['regi_date'],
    #         heartbeat=self.validated_data['heartbeat'],
    #         bp=self.validated_data['bp'],
    #         dust_allergy=self.validated_data['dust_allergy'],
    #         medication=self.validated_data['medication'],
    #     )
    #     data.save()
    #     return data


class PatientImageSerializer(serializers.ModelSerializer):

    patient = serializers.RelatedField(source='PatientDetails', read_only=True)

    class Meta:
        model = PatientImage
        fields = '__all__'


class QuestionAnswersForm1Serializer(serializers.ModelSerializer):

    class Meta:
        model = QuestionAnswersForm1
        fields = '__all__'


class QuestionAnswersForm2Serializer(serializers.ModelSerializer):

    class Meta:
        model = QuestionAnswersForm2
        fields = '__all__'


class PatientForm1Serializer(serializers.ModelSerializer):

    patient = serializers.RelatedField(source='PatientDetails', read_only=True)
    # form1 = serializers.RelatedField(source='QuestionAnswersForm1', read_only=True)
    answer_type = serializers.CharField(read_only=True)
    question = serializers.CharField(read_only=True)
    answer = serializers.CharField(read_only=True)

    class Meta:
        model = PatientForm1
        fields = '__all__'


class PatientForm2Serializer(serializers.ModelSerializer):

    patient = serializers.RelatedField(source='PatientDetails', read_only=True)
    form2_concern1 = serializers.RelatedField(source='QuestionAnswersForm2', read_only=True)
    form2_concern2 = serializers.RelatedField(source='QuestionAnswersForm2', read_only=True)
    skin_type = serializers.CharField(read_only=True)
    answer_type = serializers.CharField(read_only=True)
    question = serializers.CharField(read_only=True)
    answer = serializers.CharField(read_only=True)

    class Meta:
        model = PatientForm2
        fields = '__all__'


class PatientDiagnoseSerializer(serializers.ModelSerializer):

    patient = serializers.RelatedField(source='PatientDetails', read_only=True)

    class Meta:
        model = DiagnoseImage
        fields = '__all__'

class PatientBeforeAfterSerializer(serializers.ModelSerializer):

    patient = serializers.RelatedField(source='PatientDetails', read_only=True)

    class Meta:
        model = PatientBeforeAfter
        fields = '__all__'

class PatientMeasureImprovementSerializer(serializers.ModelSerializer):

    patient = serializers.RelatedField(source='PatientDetails', read_only=True)

    class Meta:
        model= PatientMeasureImprovement
        fields = '__all__'


# class ServiceSerializer(serializers.ModelSerializer):
    
#     class Meta:
#         model = ServiceList
#         fields = '__all__'


# class ProductSerializer(serializers.ModelSerializer):

#     class Meta:
#         model = ProductList
#         fields = '__all__'


class PatientServiceProductSerializer(serializers.ModelSerializer):

    patient = serializers.RelatedField(source="PatientDetails", read_only=True)
    service = serializers.CharField(read_only=True)
    product = serializers.CharField(read_only=True)

    class Meta:
        model = PatientServiceProduct
        fields = '__all__'