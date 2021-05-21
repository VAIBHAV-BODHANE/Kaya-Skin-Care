from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FileUploadParser, JSONParser, FormParser

from django.contrib.auth import logout, authenticate, login
from django.http import HttpResponse
from django.db.models import Q
from django.shortcuts import render
from kayaApp.models import UserRegister, PatientDetails, PatientImage, PatientForm1, PatientForm2, QuestionAnswersForm1, QuestionAnswersForm2, DiagnoseImage,PatientBeforeAfter,PatientMeasureImprovement, ConcernList, ServiceList, ProductList, PatientServiceProduct
from kayaApp.serializers import UserRegisterSerializer, LoginSerializer, PatientDetailsSerializer, PatientImageSerializer, PatientForm1Serializer, PatientForm2Serializer, QuestionAnswersForm1Serializer, QuestionAnswersForm2Serializer, PatientDiagnoseSerializer,PatientBeforeAfterSerializer,PatientMeasureImprovementSerializer, PatientServiceProductSerializer
from kayaApp.helpers import modify_input_for_multiple_files, diagnose_multiple_files, measure_improvement_multiple_files, before_after_image
from django.core.files.uploadedfile import InMemoryUploadedFile

from django.shortcuts import render
from werkzeug.utils import secure_filename
from django.conf import settings

import ast
import json
import datetime
import pytz
utc = pytz.UTC

from io import BytesIO

from PIL import Image
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
# tf.disable_v2_behavior() 
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# # Root directory of the project

# # Import Mask RCNN
# from .mrcnn import utils
# from .mrcnn import visualize
# from .mrcnn.visualize import display_images
# from .mrcnn import model as modellib
# from .mrcnn.model import log


# ROOT_DIR = os.path.abspath("../")
# # sys.path.append(ROOT_DIR)  # To find local version of the library

# from kayaApp import skin_diagnosis

# # # # Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "kayaApp/static/weights/")

# config = skin_diagnosis.SkinConfig()

# # # # Override the training configurations with a few
# # # # changes for inferencing.
# class InferenceConfig(config.__class__):
#     # Run detection on one image at a time
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#     DETECTION_MIN_CONFIDENCE = 0.6

# config = InferenceConfig()
# config.display()

# class HyperInferenceConfig(config.__class__):
#     # Run detection on one image at a time
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#     NUM_CLASSES = 2
#     DETECTION_MIN_CONFIDENCE = 0.9


# hyper_config = HyperInferenceConfig()
# hyper_config.display()


# # Device to load the neural network on.
# # Useful if you're training a model on the same
# # machine, in which case use CPU and leave the
# # GPU for training.
# DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# # Inspect the model in training or inference modes
# # values: 'inference' or 'training'
# # TODO: code for 'training' test mode not ready yet
# TEST_MODE = "inference"


# WEIGHTS_DIR = os.path.dirname(__file__)
# print(WEIGHTS_DIR)

# weights_path = WEIGHTS_DIR + '/static/weights/mask_rcnn_skin_0040.h5'
# hyper_pig_weights_path = WEIGHTS_DIR + '/static/weights/hyper_pigmentation.h5'

# global graph
# graph1 = tf.Graph()
# graph2 = tf.Graph()

# # with tf.device(DEVICE):
# # model2 = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# # print("Loading weights ", weights_path)
# with graph1.as_default():
#     session1 = tf.Session()
#     with session1.as_default():
#         model1 = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
#         model1.load_weights(weights_path, by_name=True)

# # with graph2.as_default():
# #     session2 = tf.Session()
# #     with session2.as_default():
# #         model2 = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=hyper_config)
# #         model2.load_weights(hyper_pig_weights_path, by_name=True)

# # with tf.device(DEVICE):
# #     model2 = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)


    

# def get_ax(rows=1, cols=1, size=16):
#     """Return a Matplotlib Axes array to be used in
#     all visualizations in the notebook. Provide a
#     central point to control graph sizes.
    
#     Adjust the size attribute to control how big to render images
#     """
#     _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
#     return ax

# def skin_prediction(img_name):

#     image = np.array(img_name)
# #####INCREASE CONTRAST USING PIL ##########

#     # image = Image.fromarray(image)
#     # enhancer = ImageEnhance.Contrast(image)

#     # factor = 1.5 #gives original image
#     # im_output = enhancer.enhance(factor)
#     # image = np.array(im_output)

# #####INCREASE CONTRAST USING OPENCV ##########

#     lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=0.7, tileGridSize=(8,8))
#     cl = clahe.apply(l)
#     limg = cv2.merge((cl,a,b))
#     new_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

#     # image = image[:,:,::-1]

#     # kernel_sharpening = np.array([[-1,-1,-1],
#     #                           [-1, 9,-1],
#     #                           [-1,-1,-1]])
#     #
#     # image = cv2.filter2D(image, -1, kernel_sharpening)

#     if new_img.shape[-1] == 4:
#         new_img= image[..., :3]

#     class_names=['BG', 'acne', 'pigmented scar', 'scar', 'pih', 'hyper pigmentation', 'mole', 'open pores', 'melasma']
    
#     with graph1.as_default():
#         with session1.as_default():
#             results = model1.detect(new_img, verbose=1)

#     # with graph2.as_default():
#     #     with session2.as_default():
#     #         hyper_results = model2.detect(image, verbose=1)
    

#     # Visualize results
#     # ax = get_ax(1)
#     r = results[0]
#     # hr = hyper_results[0]
    
#     # hyper_class_names=['BG', 'hyper pigmentation']

#     final =  visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                                 class_names, r['scores'],show_bbox = True)
    
#     # final =  visualize.display_instances(final, hr['rois'], hr['masks'], hr['class_ids'],
#     #                         hyper_class_names, hr['scores'],show_bbox = True)

#     # return final,r,hr
#     return final,r


# @api_view(['POST'])
# @permission_classes([AllowAny])
# def registerUser(request):
#     if request.method == 'POST':
#         serializer = UserRegisterSerializer(data=request.data)
#         data = {}
#         if serializer.is_valid():
#             user = serializer.save()
#             data['response'] = "successfully register a new user."
#             data['email'] = user.email
#             data['username'] = user.username
#             token = Token.objects.get(user=user).key
#             data['token'] = token
#         else:
#             data = serializer.errors
#         return Response(data)


class RegisterUserView(APIView):
    permission_classes = ([AllowAny])

    def post(self, request):
        serializer = UserRegisterSerializer(data=request.data)
        data = {}
        if serializer.is_valid():
            user = serializer.save()
            data['response'] = "successfully register a new user."
            data['email'] = user.email
            data['username'] = user.username
            token = Token.objects.get(user=user).key
            data['token'] = token
            return Response(data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserInfoView(APIView):
    permission_classes = ([IsAuthenticated])

    def get(self, request):
        qs = UserRegister.objects.get(pk=request.user.id)
        serializer = UserRegisterSerializer(qs)
        return Response({"resp":serializer.data}, status=status.HTTP_200_OK)


class LoginView(APIView):
    permission_classes = ([AllowAny])

    def post(self, request, *args, **kwargs):
        serializer = LoginSerializer(data=request.data)

        user = authenticate(request, username = request.data['username'], password = request.data['password'])

        if user:
            login(request, user)
            token = Token.objects.filter(user=user).first()
            if token:
                token.delete()
            token = Token.objects.create(user=user)
            user_token = token.key
            data = {
                'username': request.data['username'],
                'auth_token': user_token
            }
            return Response(data, status=status.HTTP_200_OK)
        return Response({'message': 'Username or Password Incorrect!'}, status=status.HTTP_404_NOT_FOUND)


class LogoutView(APIView):
    permission_classes = [(IsAuthenticated)]
    serailizer_class = LoginSerializer

    def post(self, request):
        token = request.auth
        print(token)
        try: 
            token = Token.objects.get(key=token).delete()
            logout(request)
        except:
            return Response({"error": "session1 does not exists."}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"message": "Successfully logged out."}, status=status.HTTP_200_OK)


class PatientView(APIView):
    permission_classes = (IsAuthenticated, )
    
    def query_set(self):
        user = self.request.user
        return user

    def get(self, request, *args, **kwargs):
        obj = self.query_set()
        print(obj.id)
        qs = PatientDetails.objects.filter(doctor=obj.id)
        print(qs)
        if len(qs)==0:
            return Response({"Message":"You have no patients!"}, status=status.HTTP_204_NO_CONTENT)
        serializer = PatientDetailsSerializer(qs, many=True)
        print(serializer)
        return Response({"resp":serializer.data},status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        serializer = PatientDetailsSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(doctor=request.user)
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PatientUpdateDelete(APIView):

    permission_classes = ([IsAuthenticated])

    def get_object(self, pk):
        try:
            return PatientDetails.objects.get(pk=pk)
        except PatientDetails.DoesNotExist:
            return status.HTTP_404_NOT_FOUND
    
    def get(self, request, pk, format=None):
        qs = self.get_object(pk)
        if qs != 404:
            # if qs.doctor.id == request.user.id:
            serializer = PatientDetailsSerializer(qs)
            return Response(serializer.data, status=status.HTTP_200_OK)
            # else:
            #     return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Patient does not exist!"}, status=status.HTTP_404_NOT_FOUND)

    def patch(self, request, pk, format=None):
        qs = self.get_object(pk)
        if qs != 404:
            # if qs.doctor.id == request.user.id:
            serializer = PatientDetailsSerializer(qs, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save(doctor=request.user)
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            # else:
            #     return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Patient does not exist"}, status=status.HTTP_404_NOT_FOUND)
    
    def delete(self, request, pk, format=None):
        qs = self.get_object(pk)
        if qs != 404:
            if qs.doctor.id == request.user.id:
                qs.delete()
                return Response({"message":"Record has been deleted!"}, status=status.HTTP_200_OK)
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Patient does not exist!"}, status=status.HTTP_404_NOT_FOUND)

    # def patch(self, request, pk, *args, **kwargs):
    #     qs = PatientDetails.objects.get(pk=pk)
    #     serializer = PatientDetailsSerializer(qs, data=request.data, partial=True)
    #     if serializer.is_valid():
    #         serializer.save()
    #         return Response(serializer.data)
    #     return Response(serializer.errors)

    # def delete(self, request, pk, *args, **kwargs):
    #     qs = PatientDetails.objects.get(pk=pk)
    #     qs.delete()
    #     return Response('Record has been deleted')


# @api_view(['GET'])
# @permission_classes([IsAuthenticated])
# def patient_list(request):
#     if request.method == 'GET':
#         qs = PatientDetails.objects.all()
#         serializer = PatientDetailsSerilizer(qs, many=True)
#         return Response(serializer.data)

class PatientNamePhoneInfoView(APIView):
    permission_classes = ([IsAuthenticated])

    def get(self, request):
        qs = PatientDetails.objects.all().order_by()
        print(qs)

        if len(qs) != 0:
            serializer = PatientDetailsSerializer(qs, many=True)
            print(serializer.data)
            data = {}
            p = 1
            for i in serializer.data:
                data['patient' + str(p)] = [
                    i['patient_name'],
                    i['phone']
                ]
                p += 1

            # data['patient_name'] = i['patient_name']
            # data['phone'] = i['phone']
            return Response({"resp": data}, status=status.HTTP_200_OK)
        else:
            return Response({"error": "There is no patient"}, status=status.HTTP_404_NOT_FOUND)


class PatientImageList(APIView):
    permission_classes = ([IsAuthenticated])
    parser_classes = [MultiPartParser, FileUploadParser, FormParser]

    def get_object(self, pk):
        try:
            return PatientDetails.objects.get(pk=pk)
        except PatientDetails.DoesNotExist:
            return status.HTTP_404_NOT_FOUND

    def get_image_object(self, pk):
        try:
            return PatientImage.objects.get(pk=pk)
        except PatientImage.DoesNotExist:
            return status.HTTP_404_NOT_FOUND

    def get(self, request, pk, *args, **kwargs):
        qs = PatientImage.objects.filter(patient=pk).order_by('updated_date').distinct('updated_date')
        print(qs)
        if len(qs)==0:
            return Response({"Message":"No Images!"}, status=status.HTTP_204_NO_CONTENT)
        elif qs[0].patient.doctor.id == request.user.id:
            lst = []
            for i in qs:
                data = {}
                print(i.updated_date)
                qs1 = PatientImage.objects.filter(patient=pk, updated_date = i.updated_date)
                data['date'] = qs1[0].updated_date
                # data['image'] = []
                serializer = PatientImageSerializer(qs1, many=True)
                data['datas'] = serializer.data
                lst.append(data)
            return Response({"resp":lst},status=status.HTTP_200_OK)
        else:
            return Response({"error": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)

    def post(self, request, pk, *args, **kwargs):
        qs = self.get_object(pk)
        print(qs)
        if qs.doctor.id == request.user.id:
            images = dict((request.data).lists())['image']
            print(images)
            flag = 1
            arr = []
            for img_name in images:
                # img = img_name['name'].split('/')
                # print(img[-1])
                # i = Image.open(img_name['name'])
                # # print(i)
                # buffer = BytesIO()
                # i.save(buffer, format='JPEG', quality=85)
                # buffer.seek(0)
                # new_pic= InMemoryUploadedFile(buffer, 'ImageField',
                #                     img[-1],
                #                     'image/jpeg',
                #                     sys.getsizeof(buffer), None)
                modified_data = modify_input_for_multiple_files(img_name)
                print(modified_data)
                serializer = PatientImageSerializer(data=modified_data)
                if serializer.is_valid():
                    serializer.save(patient=qs)
                    arr.append(serializer.data)
                else:
                    flag = 0
            if flag == 1:
                return Response(arr, status=status.HTTP_201_CREATED)
            else:
                return Response(arr, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)

    def patch(self, request, pk, *args, **kwargs):
        qs = self.get_image_object(pk)
        if qs != 404:
            if qs.patient.doctor.id == request.user.id:
                images = request.data['image']
                print(images)
                flag = 1
                arr = []
                for img_name in images:
                    # img = img_name['name'].split('/')
                    # print(img[-1])
                    # i = Image.open(img_name['name'])
                    # print(i)
                    # buffer = BytesIO()
                    # i.save(buffer, format='JPEG', quality=85)
                    # buffer.seek(0)
                    # new_pic= InMemoryUploadedFile(buffer, 'ImageField',
                    #                     img[-1],
                    #                     'image/jpeg',
                    #                     sys.getsizeof(buffer), None)
                    modified_data = modify_input_for_multiple_files(img_name)
                    print(modified_data)
                    serializer = PatientImageSerializer(data=modified_data)
                    if serializer.is_valid():
                        serializer.save(patient=qs)
                        arr.append(serializer.data)
                    else:
                        flag = 0
                if flag == 1:
                    return Response(arr, status=status.HTTP_201_CREATED)
                else:
                    return Response(arr, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error: Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Image does not exist!"}, status=status.HTTP_404_NOT_FOUND)

    def delete(self, request, pk, *args, **kwargs):
        qs = self.get_image_object(pk)
        if qs != 404:
            if qs.patient.doctor.id == request.user.id:
                qs.delete()
                return Response({"message: Record has been deleted!"}, status=status.HTTP_204_NO_CONTENT)
            else:
                return Response({"error: Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Image does not exist!"}, status=status.HTTP_404_NOT_FOUND)


class QuestionAnswer1(APIView):
    permission_classes([IsAuthenticated])

    def get(self, request, *args, **kwargs):
        qs = QuestionAnswersForm1.objects.all().order_by('id')
        serializer = QuestionAnswersForm1Serializer(qs, many=True)
        data = {}
        textfield = []
        checkbox = []
        all = []
        a = 1
        for i in serializer.data:
            for j,k in i.items():
                print(j, '------------------',k)
                if type(k) == int:
                    question_id = k
                    continue
                else:
                    if k['question'] == 'Current Medications':
                        textfield.append({
                            "id": question_id,
                            "question_type" : k
                        })
                    elif k['question'] == 'Do you have any Medical Condition?':
                        checkbox.append({
                            "id": question_id,
                            "question_type" : k
                        })
                    elif k['question'] != 'Current Medications' and k['question'] != 'Do you have any Medical Condition?':
                        all.append({
                            "id": question_id,
                            "question_type" : k
                        })
                    else:
                        lifestyle.append({
                            "id": question_id,
                            "question_type" : k
                        })
        data['textfield'] = textfield
        data['checkbox'] = checkbox
        data['resp'] = all
        return Response(data, status=status.HTTP_200_OK)
        # return Response({"resp": serializer.data}, status=status.HTTP_200_OK)




# class QuestionAnswer2(APIView):
#     permission_classes([IsAuthenticated])

#     def get(self, request, *args, **kwargs):
#         qs = QuestionAnswersForm2.objects.all()
#         serializer = QuestionAnswersForm2Serializer(qs, many=True)
#         return Response({"resp":serializer.data},status=status.HTTP_200_OK)


class QuestionAnswer2(APIView):
    permission_classes([IsAuthenticated])
    
    def get_form2(self, num_skin1, num_skin2):
        print(num_skin1, num_skin2)
        return 'hello'
    
    def get_object(self,pk):
        try:
            return PatientForm1.objects.filter(patient=pk)
        except PatientForm1.DoesNotExist:
            return status.HTTP_404_NOT_FOUND


    def get(self, request, pk, *args, **kwargs):
        qs_form1 = self.get_object(pk)
        print(qs_form1)
        if qs_form1:            #REPLACED if qs_form1 != 404 with if qs_form1
            if qs_form1[0].patient.doctor.id == request.user.id:
                oily_skin = 0
                normal_skin = 0
                combination_skin = 0
                sensitive_skin = 0
                dry_skin = 0
                for i in qs_form1:
                    # print(i.answer_type)
                    # print(i.answer_type.strip('][').replace("(", "").replace(")","").split("),"))
                    listA = i.answer_type.strip('][').replace("(", "").replace(")","").split("),")
                    b = listA[0].split(",")
                    # print(b)
                    for j in range(len(b)):
                        if j % 2 == 0:
                            continue
                        else:
                            x = b[j].replace('"', "").replace(" ", "").replace("'", "")
                            # print(x)
                            if x =='oily':
                                oily_skin += 1
                            elif x == 'normal':
                                normal_skin += 1
                            elif x == 'combination':
                                combination_skin += 1
                            elif x == 'sensitive':
                                sensitive_skin += 1
                            elif x == 'dry':
                                dry_skin += 1
                            else:
                                pass
                    # for k in i.answer_type.strip('][').replace("(", "").split("),"):
                    #     print(list(map(eval, k+')')))
                    #     print(k + ')')
                    #     a.append(tuple(k+')')
                    # print(a)

                    # for j in range(len(list(i.answer_type).split(", "))):
                    #     print(i.answer_type[j])
                        # if j[1] =='oily':
                        #     oily_skin += 1
                        # elif j[1] == 'normal':
                        #     normal_skin += 1
                        # elif j[1] == 'combination':
                        #     combination_skin += 1
                        # elif j[1] == 'sensitive':
                        #     sensitive_skin += 1
                        # elif j[1] == 'dry':
                        #     dry_skin += 1
                        # else:
                        #     pass
                
                total_answer = [oily_skin, normal_skin, combination_skin, sensitive_skin, dry_skin]
                print(total_answer)
                # li = cycle(total_answer)
                # print(li)
                # for i in cycle(total_answer):
                #     for j in (total_answer):
                #         print(i, next(li))
                        # a = total_answer.index(j)
                        
                        # print(i, next(j))
                        # print(total_answer.index(j))

                        # print(self.get_form2(i,j))
                        # return 'Hello'
                
                if oily_skin >= 3 and normal_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(Q(question_type__type__in=['oily','normal']))
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    print(serializer.data)
                    data = {}
                    question_type = {}
                    form2_id = []
                    typ = []
                    concern_id = []
                    option = []
                    all_opt = set()
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            print(i, '--------------', j)
                            if type(j) == dict:
                                question_type['question'] = j['question']
                                typ.append(j['type'])
                                
                                print('--------joption---------',j['option'])
                                check_opt =[]
                                for opt in j['option']:
                                    if opt['answer_type'] != 'none':
                                        all_opt.add(opt['answer_type'])
                                print(all_opt)
                            else:
                                form2_id.append(j)
                    
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            if type(j) == dict:
                                for o in j['option']:
                                    for opti in all_opt:
                                        if opti == o['answer_type']:
                                            if (o['answer_type'] not in check_opt):
                                                check_opt.append(o['answer_type'])
                                                print(opti, '----------------', o['answer_type'])
                                                # option.add(o['option' + str((j['option'].index(o)) + 1)])
                                                option.append({
                                                    "option"+ str((j['option'].index(o)) + 1): o['option' + str((j['option'].index(o)) + 1)],
                                                    "answer_type": opti
                                                    
                                                })
                                                break
                            
                    print('-----------------------------------------')
                    print(all_opt)
                    print('-----------------------------------------')
                    print(option)
                    print('-----------------------------------------')
                            # print(j['option'])
                    

                    data['id'] = form2_id
                    question_type['type'] = typ
                    question_type['option'] = option       
                    data['question_type'] = question_type
                    print(data)
                            

                    return Response({"resp": data, "flag": "false"})

                elif oily_skin >= 3 and combination_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(Q(question_type__type__in=['oily','combination']))
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    print(serializer.data)
                    data = {}
                    question_type = {}
                    form2_id = []
                    typ = []
                    concern_id = []
                    option = set()
                    all_opt = set()
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            print(i, '--------------', j)
                            if type(j) == dict:
                                question_type['question'] = j['question']
                                typ.append(j['type'])
                                
                                print('--------joption---------',j['option'])
                                check_opt =[]
                                for opt in j['option']:
                                    if opt['answer_type'] != 'none':
                                        all_opt.add(opt['answer_type'])
                                print(all_opt)
                            else:
                                form2_id.append(j)
                    
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            if type(j) == dict:
                                for o in j['option']:
                                    for opti in all_opt:
                                        if opti == o['answer_type']:
                                            if (o['answer_type'] not in check_opt):
                                                check_opt.append(o['answer_type'])
                                                print(opti, '----------------', o['answer_type'])
                                                # option.add(o['option' + str((j['option'].index(o)) + 1)])
                                                option.append({
                                                    "option"+ str((j['option'].index(o)) + 1): o['option' + str((j['option'].index(o)) + 1)],
                                                    "answer_type": opti
                                                    
                                                })
                                                break
                            
                    print('-----------------------------------------')
                    print(all_opt)
                    print('-----------------------------------------')
                    print(option)
                    print('-----------------------------------------')
                            # print(j['option'])
                            

                    data['id'] = form2_id
                    question_type['type'] = typ
                    question_type['option'] = option       
                    data['question_type'] = question_type
                    print(data)
                            

                    return Response({"resp": data, "flag": "false"})

                elif oily_skin >= 3 and sensitive_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(Q(question_type__type__in=['oily','sensitive']))
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    print(serializer.data)
                    data = {}
                    question_type = {}
                    form2_id = []
                    typ = []
                    concern_id = []
                    option = set()
                    all_opt = set()
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            print(i, '--------------', j)
                            if type(j) == dict:
                                question_type['question'] = j['question']
                                typ.append(j['type'])
                                
                                print('--------joption---------',j['option'])
                                check_opt =[]
                                for opt in j['option']:
                                    if opt['answer_type'] != 'none':
                                        all_opt.add(opt['answer_type'])
                                print(all_opt)
                            else:
                                form2_id.append(j)
                    
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            if type(j) == dict:
                                for o in j['option']:
                                    for opti in all_opt:
                                        if opti == o['answer_type']:
                                            if (o['answer_type'] not in check_opt):
                                                check_opt.append(o['answer_type'])
                                                print(opti, '----------------', o['answer_type'])
                                                # option.add(o['option' + str((j['option'].index(o)) + 1)])
                                                option.append({
                                                    "option"+ str((j['option'].index(o)) + 1): o['option' + str((j['option'].index(o)) + 1)],
                                                    "answer_type": opti
                                                    
                                                })
                                                break
                            
                    print('-----------------------------------------')
                    print(all_opt)
                    print('-----------------------------------------')
                    print(option)
                    print('-----------------------------------------')
                            # print(j['option'])
                            

                    data['id'] = form2_id
                    question_type['type'] = typ
                    question_type['option'] = option       
                    data['question_type'] = question_type
                    print(data)
                            

                    return Response({"resp": data, "flag": "false"})

                elif oily_skin >= 3 and dry_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(Q(question_type__type__in=['oily','combination']))
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    print(serializer.data)
                    data = {}
                    question_type = {}
                    form2_id = []
                    typ = []
                    concern_id = []
                    option = set()
                    all_opt = set()
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            print(i, '--------------', j)
                            if type(j) == dict:
                                question_type['question'] = j['question']
                                typ.append(j['type'])
                                
                                print('--------joption---------',j['option'])
                                check_opt =[]
                                for opt in j['option']:
                                    if opt['answer_type'] != 'none':
                                        all_opt.add(opt['answer_type'])
                                print(all_opt)
                            else:
                                form2_id.append(j)
                    
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            if type(j) == dict:
                                for o in j['option']:
                                    for opti in all_opt:
                                        if opti == o['answer_type']:
                                            if (o['answer_type'] not in check_opt):
                                                check_opt.append(o['answer_type'])
                                                print(opti, '----------------', o['answer_type'])
                                                # option.add(o['option' + str((j['option'].index(o)) + 1)])
                                                option.append({
                                                    "option"+ str((j['option'].index(o)) + 1): o['option' + str((j['option'].index(o)) + 1)],
                                                    "answer_type": opti
                                                    
                                                })
                                                break
                            
                    print('-----------------------------------------')
                    print(all_opt)
                    print('-----------------------------------------')
                    print(option)
                    print('-----------------------------------------')
                            # print(j['option'])
                            

                    data['id'] = form2_id
                    question_type['type'] = typ
                    question_type['option'] = option       
                    data['question_type'] = question_type
                    print(data)
                            

                    return Response({"resp": data, "flag": "false"})

                elif normal_skin >= 3 and combination_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(Q(question_type__type__in=['normal','combination']))
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    print(serializer.data)
                    data = {}
                    question_type = {}
                    form2_id = []
                    typ = []
                    concern_id = []
                    option = set()
                    all_opt = set()
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            print(i, '--------------', j)
                            if type(j) == dict:
                                question_type['question'] = j['question']
                                typ.append(j['type'])
                                
                                print('--------joption---------',j['option'])
                                check_opt =[]
                                for opt in j['option']:
                                    if opt['answer_type'] != 'none':
                                        all_opt.add(opt['answer_type'])
                                print(all_opt)
                            else:
                                form2_id.append(j)
                    
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            if type(j) == dict:
                                for o in j['option']:
                                    for opti in all_opt:
                                        if opti == o['answer_type']:
                                            if (o['answer_type'] not in check_opt):
                                                check_opt.append(o['answer_type'])
                                                print(opti, '----------------', o['answer_type'])
                                                # option.add(o['option' + str((j['option'].index(o)) + 1)])
                                                option.append({
                                                    "option"+ str((j['option'].index(o)) + 1): o['option' + str((j['option'].index(o)) + 1)],
                                                    "answer_type": opti
                                                    
                                                })
                                                break
                            
                    print('-----------------------------------------')
                    print(all_opt)
                    print('-----------------------------------------')
                    print(option)
                    print('-----------------------------------------')
                            # print(j['option'])
                            

                    data['id'] = form2_id
                    question_type['type'] = typ
                    question_type['option'] = option       
                    data['question_type'] = question_type
                    print(data)
                            

                    return Response({"resp": data, "flag": "false"})

                elif normal_skin >= 3 and sensitive_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(Q(question_type__type__in=['normal','sensitive']))
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    print(serializer.data)
                    data = {}
                    question_type = {}
                    form2_id = []
                    typ = []
                    concern_id = []
                    option = set()
                    all_opt = set()
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            print(i, '--------------', j)
                            if type(j) == dict:
                                question_type['question'] = j['question']
                                typ.append(j['type'])
                                
                                print('--------joption---------',j['option'])
                                check_opt =[]
                                for opt in j['option']:
                                    if opt['answer_type'] != 'none':
                                        all_opt.add(opt['answer_type'])
                                print(all_opt)
                            else:
                                form2_id.append(j)
                    
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            if type(j) == dict:
                                for o in j['option']:
                                    for opti in all_opt:
                                        if opti == o['answer_type']:
                                            if (o['answer_type'] not in check_opt):
                                                check_opt.append(o['answer_type'])
                                                print(opti, '----------------', o['answer_type'])
                                                # option.add(o['option' + str((j['option'].index(o)) + 1)])
                                                option.append({
                                                    "option"+ str((j['option'].index(o)) + 1): o['option' + str((j['option'].index(o)) + 1)],
                                                    "answer_type": opti
                                                    
                                                })
                                                break
                            
                    print('-----------------------------------------')
                    print(all_opt)
                    print('-----------------------------------------')
                    print(option)
                    print('-----------------------------------------')
                            # print(j['option'])
                            

                    data['id'] = form2_id
                    question_type['type'] = typ
                    question_type['option'] = option       
                    data['question_type'] = question_type
                    print(data)
                            

                    return Response({"resp": data, "flag": "false"})

                elif normal_skin >= 3 and dry_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(Q(question_type__type__in=['normal','dry']))
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    print(serializer.data)
                    data = {}
                    question_type = {}
                    form2_id = []
                    typ = []
                    concern_id = []
                    option = set()
                    all_opt = set()
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            print(i, '--------------', j)
                            if type(j) == dict:
                                question_type['question'] = j['question']
                                typ.append(j['type'])
                                
                                print('--------joption---------',j['option'])
                                check_opt =[]
                                for opt in j['option']:
                                    if opt['answer_type'] != 'none':
                                        all_opt.add(opt['answer_type'])
                                print(all_opt)
                            else:
                                form2_id.append(j)
                    
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            if type(j) == dict:
                                for o in j['option']:
                                    for opti in all_opt:
                                        if opti == o['answer_type']:
                                            if (o['answer_type'] not in check_opt):
                                                check_opt.append(o['answer_type'])
                                                print(opti, '----------------', o['answer_type'])
                                                # option.add(o['option' + str((j['option'].index(o)) + 1)])
                                                option.append({
                                                    "option"+ str((j['option'].index(o)) + 1): o['option' + str((j['option'].index(o)) + 1)],
                                                    "answer_type": opti
                                                    
                                                })
                                                break
                            
                    print('-----------------------------------------')
                    print(all_opt)
                    print('-----------------------------------------')
                    print(option)
                    print('-----------------------------------------')
                            # print(j['option'])
                            

                    data['id'] = form2_id
                    question_type['type'] = typ
                    question_type['option'] = option       
                    data['question_type'] = question_type
                    print(data)
                            

                    return Response({"resp": data, "flag": "false"})

                elif combination_skin >= 3 and sensitive_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(Q(question_type__type__in=['combination','sensitive']))
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    print(serializer.data)
                    data = {}
                    question_type = {}
                    form2_id = []
                    typ = []
                    concern_id = []
                    option = set()
                    all_opt = set()
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            print(i, '--------------', j)
                            if type(j) == dict:
                                question_type['question'] = j['question']
                                typ.append(j['type'])
                                
                                print('--------joption---------',j['option'])
                                check_opt =[]
                                for opt in j['option']:
                                    if opt['answer_type'] != 'none':
                                        all_opt.add(opt['answer_type'])
                                print(all_opt)
                            else:
                                form2_id.append(j)
                    
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            if type(j) == dict:
                                for o in j['option']:
                                    for opti in all_opt:
                                        if opti == o['answer_type']:
                                            if (o['answer_type'] not in check_opt):
                                                check_opt.append(o['answer_type'])
                                                print(opti, '----------------', o['answer_type'])
                                                # option.add(o['option' + str((j['option'].index(o)) + 1)])
                                                option.append({
                                                    "option"+ str((j['option'].index(o)) + 1): o['option' + str((j['option'].index(o)) + 1)],
                                                    "answer_type": opti
                                                    
                                                })
                                                break
                            
                    print('-----------------------------------------')
                    print(all_opt)
                    print('-----------------------------------------')
                    print(option)
                    print('-----------------------------------------')
                            # print(j['option'])
                            

                    data['id'] = form2_id
                    question_type['type'] = typ
                    question_type['option'] = option       
                    data['question_type'] = question_type
                    print(data)
                            

                    return Response({"resp": data, "flag": "false"})

                elif combination_skin >= 3 and dry_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(Q(question_type__type__in=['combination','dry']))
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    print(serializer.data)
                    data = {}
                    question_type = {}
                    form2_id = []
                    typ = []
                    concern_id = []
                    option = set()
                    all_opt = set()
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            print(i, '--------------', j)
                            if type(j) == dict:
                                question_type['question'] = j['question']
                                typ.append(j['type'])
                                
                                print('--------joption---------',j['option'])
                                check_opt =[]
                                for opt in j['option']:
                                    if opt['answer_type'] != 'none':
                                        all_opt.add(opt['answer_type'])
                                print(all_opt)
                            else:
                                form2_id.append(j)
                    
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            if type(j) == dict:
                                for o in j['option']:
                                    for opti in all_opt:
                                        if opti == o['answer_type']:
                                            if (o['answer_type'] not in check_opt):
                                                check_opt.append(o['answer_type'])
                                                print(opti, '----------------', o['answer_type'])
                                                # option.add(o['option' + str((j['option'].index(o)) + 1)])
                                                option.append({
                                                    "option"+ str((j['option'].index(o)) + 1): o['option' + str((j['option'].index(o)) + 1)],
                                                    "answer_type": opti
                                                    
                                                })
                                                break
                            
                    print('-----------------------------------------')
                    print(all_opt)
                    print('-----------------------------------------')
                    print(option)
                    print('-----------------------------------------')
                            # print(j['option'])
                            

                    data['id'] = form2_id
                    question_type['type'] = typ
                    question_type['option'] = option       
                    data['question_type'] = question_type
                    print(data)
                            

                    return Response({"resp": data, "flag": "false"})

                elif sensitive_skin >= 3 and dry_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(Q(question_type__type__in=['sensitive','dry']))
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    print(serializer.data)
                    data = {}
                    question_type = {}
                    form2_id = []
                    typ = []
                    concern_id = []
                    option = set()
                    all_opt = set()
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            print(i, '--------------', j)
                            if type(j) == dict:
                                question_type['question'] = j['question']
                                typ.append(j['type'])
                                
                                print('--------joption---------',j['option'])
                                check_opt =[]
                                for opt in j['option']:
                                    if opt['answer_type'] != 'none':
                                        all_opt.add(opt['answer_type'])
                                print(all_opt)
                            else:
                                form2_id.append(j)
                    
                    for d in serializer.data:
                        # print('concern---------',concern.concern)
                        for i, j in d.items():
                            if type(j) == dict:
                                for o in j['option']:
                                    for opti in all_opt:
                                        if opti == o['answer_type']:
                                            if (o['answer_type'] not in check_opt):
                                                check_opt.append(o['answer_type'])
                                                print(opti, '----------------', o['answer_type'])
                                                # option.add(o['option' + str((j['option'].index(o)) + 1)])
                                                option.append({
                                                    "option"+ str((j['option'].index(o)) + 1): o['option' + str((j['option'].index(o)) + 1)],
                                                    "answer_type": opti
                                                    
                                                })
                                                break
                            
                    print('-----------------------------------------')
                    print(all_opt)
                    print('-----------------------------------------')
                    print(option)
                    print('-----------------------------------------')
                            # print(j['option'])
                            

                    data['id'] = form2_id
                    question_type['type'] = typ
                    question_type['option'] = option       
                    data['question_type'] = question_type
                    print(data)
                            

                    return Response({"resp": data, "flag": "false"})

                elif oily_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(question_type__type='oily')
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    data = {}
                    id = []
                    question_type = {}
                    typ = []
                    for i in serializer.data:
                        id.append(i['id'])
                        typ.append(i['question_type']['type'])
                        question_type['option']= i['question_type']['option']
                        question_type['question'] = i['question_type']['question']

                    data['id'] = id
                    question_type['type'] = typ

                    data['question_type'] = question_type
                        
                    return Response({"resp": [data], "flag": "false"})

                elif normal_skin >=3:
                    qs = QuestionAnswersForm2.objects.filter(question_type__type='normal')
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    data = {}
                    id = []
                    question_type = {}
                    typ = []
                    for i in serializer.data:
                        id.append(i['id'])
                        typ.append(i['question_type']['type'])
                        question_type['option']= i['question_type']['option']
                        question_type['question'] = i['question_type']['question']

                    data['id'] = id
                    question_type['type'] = typ

                    data['question_type'] = question_type
                        
                    return Response({"resp": [data], "flag": "false"})

                elif combination_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(question_type__type='combination')
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    data = {}
                    id = []
                    question_type = {}
                    typ = []
                    for i in serializer.data:
                        id.append(i['id'])
                        typ.append(i['question_type']['type'])
                        question_type['option']= i['question_type']['option']
                        question_type['question'] = i['question_type']['question']

                    data['id'] = id
                    question_type['type'] = typ

                    data['question_type'] = question_type
                        
                    return Response({"resp": [data], "flag": "false"})

                elif sensitive_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(question_type__type='sensitive')
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    data = {}
                    id = []
                    question_type = {}
                    typ = []
                    for i in serializer.data:
                        id.append(i['id'])
                        typ.append(i['question_type']['type'])
                        question_type['option']= i['question_type']['option']
                        question_type['question'] = i['question_type']['question']

                    data['id'] = id
                    question_type['type'] = typ

                    data['question_type'] = question_type
                        
                    return Response({"resp": [data], "flag": "false"})

                elif dry_skin >= 3:
                    qs = QuestionAnswersForm2.objects.filter(question_type__type='dry')
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    data = {}
                    id = []
                    question_type = {}
                    typ = []
                    for i in serializer.data:
                        id.append(i['id'])
                        typ.append(i['question_type']['type'])
                        question_type['option']= i['question_type']['option']
                        question_type['question'] = i['question_type']['question']

                    data['id'] = id
                    question_type['type'] = typ

                    data['question_type'] = question_type
                        
                    return Response({"resp": [data], "flag": "false"})

                else:
                    qs = QuestionAnswersForm2.objects.all()
                    serializer = QuestionAnswersForm2Serializer(qs, many=True)
                    data = {}
                    id = []
                    question_type = {}
                    typ = []
                    for i in serializer.data:
                        id.append(i['id'])
                        typ.append(i['question_type']['type'])
                        question_type['option']= i['question_type']['option']
                        question_type['question'] = i['question_type']['question']

                    data['id'] = id
                    question_type['type'] = typ

                    data['question_type'] = question_type
                        
                    return Response({"resp": [data], "flag": "false"})
            else:
                return Response({"error": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"message": "Form1 is not added"}, status=status.HTTP_400_BAD_REQUEST)
        


class PatientForm1List(APIView):
    permission_classes = ([IsAuthenticated])

    def get(self, request, pk, *args, **kwargs):
        # print(pk)
        qs = PatientForm1.objects.filter(patient=pk)
        

        if len(qs) == 0:
            return Response(status = status.HTTP_204_NO_CONTENT)
        elif qs[0].patient.doctor.id == request.user.id:
            data = {}
            
            serializer = PatientForm1Serializer(qs, many=True)
            # print(serializer.data[0]['answer_type'])
            data['id'] = serializer.data[0]['id']
            data['flag'] = True
            ques_answer = []
            print(type(serializer.data[0]['question']))
            ques = ast.literal_eval(serializer.data[0]['answer_type'])
            # print(ques[0][0])
            for i, j, k in zip(ast.literal_eval(serializer.data[0]['answer_type']), ast.literal_eval(serializer.data[0]['question']), ast.literal_eval(serializer.data[0]['answer'])):
                if type(k) == list:
                    k = str(k)
                ques_answer.append({
                    "serial_no": i[0],
                    "question": j,
                    "answer": k
                })
            data['question&answers'] = ques_answer
            # data = {}
            # for i in qs:
            #     print(i.form1.question_options['type'])
            #     # return Response({"resp":[{"basic":"g"}]},status=status.HTTP_200_OK)
            return Response({"resp": data},status=status.HTTP_200_OK)
        else:
            return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)

    def post(self, request, pk, *args, **kwargs):
        qs1 = PatientDetails.objects.get(pk=pk)
        qs2 = QuestionAnswersForm1.objects.all().order_by('id')
        try:
            qs3 = PatientForm1.objects.get(patient=pk)
        except PatientForm1.DoesNotExist:
            qs3 = status.HTTP_404_NOT_FOUND
        # print(request.data)
        answer_type = []
        question_list = []
        answer_list = []
        checkbox = request.data['checkbox']
        print('checkbox---------',checkbox)
        textfield = request.data['textfield']
        print('textfield---------',textfield)
        question = []
        answer =[]
        for c in range(len(checkbox)):
            for i, k in zip(checkbox[c].items(), textfield[c].items()):
                print(i, k)
                if i[0].lower() == k[0].lower() == 'question' :
                    question.append(i[1])
                    question.append(k[1])
                elif i[0].lower() == k[0].lower() == 'answer':
                    answer.append(i[1])
                    answer.append(k[1])
        for q in request.data['question']:
            question.append(q)
        for a in request.data['answer']:
            answer.append(a)
                
        question.insert(10, question.pop(1))
        answer.insert(10, answer.pop(1))
        print(question)
        print(answer)
        if qs1.doctor.id == request.user.id:
            if qs3 == 404:
                for question_answer_list, question, answer in zip(qs2, question, answer):
                    for i in qs2:
                        if question == i.question_type['question']:
                            if answer == i.question_type['question_options']['option'][0]['option1']['text'][0]['data1']:
                                answer_type.append((i.id,i.question_type['question_options']['option'][0]['option1']['text'][1]['data2']))
                                question_list.append(i.question_type['question'])
                                answer_list.append(i.question_type['question_options']['option'][0]['option1']['text'][0]['data1'])

                            elif answer == i.question_type['question_options']['option'][1]['option2']['text'][0]['data1']:
                                answer_type.append((i.id,i.question_type['question_options']['option'][1]['option2']['text'][1]['data2']))
                                question_list.append(i.question_type['question'])
                                answer_list.append(i.question_type['question_options']['option'][1]['option2']['text'][0]['data1'])

                            elif answer == i.question_type['question_options']['option'][2]['option3']['text'][0]['data1']:
                                answer_type.append((i.id,i.question_type['question_options']['option'][2]['option3']['text'][1]['data2']))
                                question_list.append(i.question_type['question'])
                                answer_list.append(i.question_type['question_options']['option'][2]['option3']['text'][0]['data1'])

                            elif answer == i.question_type['question_options']['option'][3]['option4']['text'][0]['data1']:
                                answer_type.append((i.id,i.question_type['question_options']['option'][3]['option4']['text'][1]['data2']))
                                question_list.append(i.question_type['question'])
                                answer_list.append(i.question_type['question_options']['option'][3]['option4']['text'][0]['data1'])

                            elif answer == i.question_type['question_options']['option'][4]['option5']['text'][0]['data1']:
                                answer_type.append((i.id,i.question_type['question_options']['option'][4]['option5']['text'][1]['data2']))
                                question_list.append(i.question_type['question'])
                                answer_list.append(i.question_type['question_options']['option'][4]['option5']['text'][0]['data1'])
                            else:
                                answer_type.append((i.id,'None'))
                                question_list.append(i.question_type['question'])
                                answer_list.append(answer)
                                

                # print(answer_type)
                # print(question_list)
                # print(answer_list)
            
                serializer = PatientForm1Serializer(data=request.data)
                if serializer.is_valid():
                    serializer.save(patient=qs1, question=question_list, answer=answer_list, answer_type=answer_type)
                    return Response(serializer.data, status=status.HTTP_200_OK)
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"message": "Form1 is already stored for this patient"}, status=status.HTTP_200_OK)
        else:
            return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
    
    def patch(self, request, pk):
        try:
            qs1 = PatientForm1.objects.get(patient=pk)
        except PatientForm1.DoesNotExist:
            return Response(status = status.HTTP_204_NO_CONTENT)

        # print(qs1.form1.id)
        qs2 = QuestionAnswersForm1.objects.all().order_by('id')
        # print(qs2)
        # print(request.data)
        answer_type = []
        question_list = []
        answer_list = []
        checkbox = request.data['checkbox']
        print('checkbox---------',checkbox)
        textfield = request.data['textfield']
        print('textfield---------',textfield)
        question = []
        answer =[]
        for c in range(len(checkbox)):
            for i, k in zip(checkbox[c].items(), textfield[c].items()):
                print(i, k)
                if i[0].lower() == k[0].lower() == 'question' :
                    question.append(i[1])
                    question.append(k[1])
                elif i[0].lower() == k[0].lower() == 'answer':
                    answer.append(i[1])
                    answer.append(k[1])
        for q in request.data['question']:
            question.append(q)
        for a in request.data['answer']:
            answer.append(a)
                
        question.insert(10, question.pop(1))
        answer.insert(10, answer.pop(1))
        print(question)
        print(answer)

        for question_answer_list, question, answer in zip(qs2, question, answer):
            print(question_answer_list, question, answer)
            for i in qs2:
                # print(i)
                if question == i.question_type['question']:
                    if answer == i.question_type['question_options']['option'][0]['option1']['text'][0]['data1']:
                        answer_type.append((i.id,i.question_type['question_options']['option'][0]['option1']['text'][1]['data2']))
                        question_list.append(i.question_type['question'])
                        answer_list.append(i.question_type['question_options']['option'][0]['option1']['text'][0]['data1'])

                    elif answer == i.question_type['question_options']['option'][1]['option2']['text'][0]['data1']:
                        answer_type.append((i.id,i.question_type['question_options']['option'][1]['option2']['text'][1]['data2']))
                        question_list.append(i.question_type['question'])
                        answer_list.append(i.question_type['question_options']['option'][1]['option2']['text'][0]['data1'])

                    elif answer == i.question_type['question_options']['option'][2]['option3']['text'][0]['data1']:
                        answer_type.append((i.id,i.question_type['question_options']['option'][2]['option3']['text'][1]['data2']))
                        question_list.append(i.question_type['question'])
                        answer_list.append(i.question_type['question_options']['option'][2]['option3']['text'][0]['data1'])

                    elif answer == i.question_type['question_options']['option'][3]['option4']['text'][0]['data1']:
                        answer_type.append((i.id,i.question_type['question_options']['option'][3]['option4']['text'][1]['data2']))
                        question_list.append(i.question_type['question'])
                        answer_list.append(i.question_type['question_options']['option'][3]['option4']['text'][0]['data1'])

                    elif answer == i.question_type['question_options']['option'][4]['option5']['text'][0]['data1']:
                        answer_type.append((i.id,i.question_type['question_options']['option'][4]['option5']['text'][1]['data2']))
                        question_list.append(i.question_type['question'])
                        answer_list.append(i.question_type['question_options']['option'][4]['option5']['text'][0]['data1'])
                    else:
                        answer_type.append((i.id,'None'))
                        question_list.append(i.question_type['question'])
                        answer_list.append(answer)

        # print(answer_type)
        # print(question_list)
        # print(answer_list)               
        if qs1.patient.doctor.id == request.user.id:
            serializer = PatientForm1Serializer(qs1, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save(question=question_list, answer=answer_list, answer_type=answer_type)
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors)
        else:
            return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        # except Exception as e:
        #     return Response({"error": f"{str(e)}"})


class PatientForm2List(APIView):
    permission_classes = ([IsAuthenticated])
    
    def get(self, request, pk, *args, **kwargs):
        qs = PatientForm2.objects.filter(patient=pk)
        if len(qs) == 0:
            return Response(status = status.HTTP_204_NO_CONTENT)
        elif qs[0].patient.doctor.id == request.user.id:
            serializer = PatientForm2Serializer(qs, many=True)
            print(serializer.data)
            data = {}
            data['id'] = serializer.data[0]['id']
            data['question'] = serializer.data[0]['question']
            data['answer'] = []
            for i in ast.literal_eval(serializer.data[0]['answer']):
                data['answer'].append(i)
            return Response({"resp": data},status=status.HTTP_200_OK)
        else:
            return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)

    def post(self, request, fk1, pk, *args, **kwargs):
        print(request.parser_context['kwargs'])
        if len(request.parser_context['kwargs']) == 3:
            fk2 = request.parser_context['kwargs']['fk2']
        else:
            fk2 = None
        qs1 = PatientDetails.objects.get(pk=pk)
        if qs1.doctor.id == request.user.id:
            qs2 = QuestionAnswersForm2.objects.filter(pk=fk1).first()
            print(qs2.question_type['type'])
            if fk2 !=None:
                qs4 = QuestionAnswersForm2.objects.filter(pk=fk2).first()
                print(qs4.question_type['type'])
            try:
                qs3 = PatientForm2.objects.get(patient=pk)
            except PatientForm2.DoesNotExist:
                qs3 = status.HTTP_404_NOT_FOUND
            if fk2 != None:
                skin_type = [qs2.question_type['type'], qs4.question_type['type']]
            else:
                skin_type = [qs2.question_type['type']]
            print(skin_type)
            print(request.data['question'])
            print(request.data['answer'])
            ans_typ=set()
            num = 1
            if fk2 != None:
                for i in request.data['answer']:
                    # for j, k in zip(qs2.question_type['option'], len(qs2.question_type['option'])):
                    for j in qs2.question_type['option']:
                        print(j)
                        if i == j['option' + str(num)]:
                            ans_typ.add(j['answer_type'])
                        num += 1
                    num = 1
                for i in request.data['answer']:
                    # for j, k in zip(qs2.question_type['option'], len(qs2.question_type['option'])):
                    for j in qs4.question_type['option']:
                        print(j)
                        if i == j['option' + str(num)]:
                            ans_typ.add(j['answer_type'])
                        num += 1
                    num = 1
                print(ans_typ)
            else:
                for i in request.data['answer']:
                    # for j, k in zip(qs2.question_type['option'], len(qs2.question_type['option'])):
                    for j in qs2.question_type['option']:
                        print(j)
                        if i == j['option' + str(num)]:
                            ans_typ.add(j['answer_type'])
                        num += 1
                    num = 1


            if qs3 == 404:
                serializer = PatientForm2Serializer(data=request.data)
                if serializer.is_valid():
                    if fk2 != None:
                        serializer.save(patient=qs1, form2_concern1=qs2, form2_concern2=qs4, answer_type = str(list(ans_typ)),question=request.data['question'], answer=request.data['answer'], skin_type=str(skin_type))
                    else:
                        serializer.save(patient=qs1, form2_concern1=qs2, answer_type = str(list(ans_typ)),question=request.data['question'], answer=request.data['answer'], skin_type=str(skin_type))
                    return Response(serializer.data, status=status.HTTP_200_OK)
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"message": "Form2 is already stored for this patient"}, status=status.HTTP_200_OK)
        else:
            return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
    
    def patch(self, request, pk):
        try:
            qs = PatientForm2.objects.get(patient=pk)
        except PatientForm2.DoesNotExist:
            return Response(status = status.HTTP_204_NO_CONTENT)
        if qs.patient.doctor.id == request.user.id:
            serializer = PatientForm2Serializer(qs, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save(question=request.data['question'], answer=request.data['answer'])
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)


class PatientDiagnoseImage(APIView):
    permission_classes = ([IsAuthenticated])
    parser_classes = [MultiPartParser, FileUploadParser, FormParser]

    def get_object(self, pk):
        try:
            return PatientDetails.objects.get(pk=pk)
        except PatientDetails.DoesNotExist:
            return status.HTTP_404_NOT_FOUND

    def get_image_object(self, pk):
        try:
            return DiagnoseImage.objects.get(pk=pk)
        except PatientImage.DoesNotExist:
            return status.HTTP_404_NOT_FOUND

    def get(self, request, pk):
        qs = DiagnoseImage.objects.filter(patient=pk)
        print(len(qs))
        if len(qs)==0:
            return Response({"Message":"No Images!"}, status=status.HTTP_204_NO_CONTENT)
        elif qs[0].patient.doctor.id == request.user.id:
            serializer = PatientDiagnoseSerializer(qs, many=True)
            return Response({"resp": serializer.data}, status=status.HTTP_200_OK)
        else:
            return Response({"message": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)

    def post(self, request, pk):
        print(self.parser_classes)
        qs = self.get_object(pk)
        if qs != 404:
            print(qs)
            print(ROOT_DIR)
            dir_path = os.path.dirname(__file__)
            print(dir_path)
            img_path = dir_path + '/static/kayaApp/images'
            print(img_path)
            # img = random.choice(os.listdir(dir_path + '\\static\\kayaApp\\images\\'))
            # image_file = os.path.join(img_path, img)
            # i = Image.open(image_file)
            


            if qs.doctor.id == request.user.id:
                if len(request.query_params) == 0:
                    images = dict((request.data).lists())['d_image']
                else:
                    images = []
                    for i,j in request.query_params.items():
                        print(i,j)
                        qs2 = PatientImage.objects.get(pk=j)
                        images.append(qs2.image)

                print(images)
                flag = 1
                arr = []

                for img_name in images:
                    # u_img = img_name['name'].split('/')
                    # print(img)
                    # image_file = os.path.join(img_path, img)
                    # print(image_file)
                    # im1 = Image.open(img_name['name'])
                    im2 = Image.open(img_name)
                    # final_image = im2.copy()

                    final_image,_ = skin_prediction(im2)
                    

                    final_image = Image.fromarray(final_image)

                    buffer1 = BytesIO()
                    buffer2 = BytesIO()
                    try:
                        final_image.save(buffer1, format='JPEG', quality=85)
                    except:
                        final_image.save(buffer1, format='PNG', quality=85)
                    try:
                        im2.save(buffer2, format='JPEG', quality=85)
                    except:
                        im2.save(buffer2, format='PNG', quality=85)
                    buffer1.seek(0)
                    buffer2.seek(0)
                    processed_pic= InMemoryUploadedFile(buffer1, 'ImageField',
                                        "mask.jpg",
                                        'image/jpeg',
                                        sys.getsizeof(buffer1), None)
                    uploaded_pic= InMemoryUploadedFile(buffer2, 'ImageField',
                                        img_name.name,
                                        'image/jpeg',
                                        sys.getsizeof(buffer2), None)

                    modified_data = diagnose_multiple_files(uploaded_pic,processed_pic)
                    print('----here--', modified_data)
                    serializer = PatientDiagnoseSerializer(data=modified_data)
                    print(serializer.is_valid())
                    if serializer.is_valid():
                        serializer.save(patient=qs)
                        arr.append(serializer.data)
                    else:
                        flag = 0
                    # data = DiagnoseImage(patient=qs, d_image=modified_data['d_image'], process_image=modified_data['process_image'])
                    # if data:
                    #     data.save()
                    #     print(data.id)
                    #     arr.append(
                    #         {
                    #             "id": data.id,
                    #             "d_image": data.d_image,
                    #             "process_image": data.process_image,
                    #             "created_date": data.created_date,
                    #             "updated_date": data.updated_date
                    #         }
                    #     )
                    # else:
                    #     flag = 0
                if flag == 1:

                    print(arr)
                    
                    return Response(arr)
                else:
                    return Response(arr, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Patient does not exist!"}, status=status.HTTP_404_NOT_FOUND)

    def patch(self, request, pk):
        qs = self.get_image_object(pk)
        dir_path = os.path.dirname(__file__)
        print(dir_path)
        img_path = dir_path + '/static/kayaApp/images'
        print(img_path)
        if qs != 404:
            if qs.patient.doctor.id == request.user.id:
                if len(request.query_params) == 0:
                    images = request.data['d_image']
                else:
                    # images = []
                    for i,j in request.query_params.items():
                        print(i,j)
                        qs2 = PatientImage.objects.get(pk=j)
                        images = qs2.image

                print(images)
                flag = 1
                arr = []

                # for img_name in images:
                # u_img = img_name['name'].split('/')
                # print(img)
                # image_file = os.path.join(img_path, img)
                # print(image_file)
                # im1 = Image.open(img_name['name'])
                im2 = Image.open(images)
                # final_image = im2.copy()

                final_image,_ = skin_prediction(im2)
                

                final_image = Image.fromarray(final_image)

                buffer1 = BytesIO()
                buffer2 = BytesIO()
                # final_image.save(buffer1, format='JPEG', quality=85)
                try:
                    final_image.save(buffer1, format='JPEG', quality=85)
                    im2.save(buffer2, format='JPEG', quality=85)
                except:
                    final_image.save(buffer1, format='PNG', quality=85)
                    im2.save(buffer2, format='PNG', quality=85)
                buffer1.seek(0)
                buffer2.seek(0)
                processed_pic= InMemoryUploadedFile(buffer1, 'ImageField',
                                    "mask.jpg",
                                    'image/jpeg',
                                    sys.getsizeof(buffer1), None)
                uploaded_pic= InMemoryUploadedFile(buffer2, 'ImageField',
                                    images.name,
                                    'image/jpeg',
                                    sys.getsizeof(buffer2), None)

                modified_data = diagnose_multiple_files(uploaded_pic,processed_pic)
                print('----here--', modified_data)
                serializer = PatientDiagnoseSerializer(qs,data=modified_data,partial = True)
                print(serializer.is_valid())
                if serializer.is_valid():
                    serializer.save()
                    arr.append(serializer.data)
                else:
                    flag = 0
                # data = DiagnoseImage(patient=qs, d_image=modified_data['d_image'], process_image=modified_data['process_image'])
                # if data:
                #     data.save()
                #     print(data.id)
                #     arr.append(
                #         {
                #             "id": data.id,
                #             "d_image": data.d_image,
                #             "process_image": data.process_image,
                #             "created_date": data.created_date,
                #             "updated_date": data.updated_date
                #         }
                #     )
                # else:
                #     flag = 0
                if flag == 1:

                    print(arr)
                    
                    return Response(arr)
                else:
                    return Response(arr, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Patient does not exist!"}, status=status.HTTP_404_NOT_FOUND)

    def delete(self, request, pk):
        qs=self.get_image_object(pk)
        qs.delete()
        return Response({"message":"Record has been deleted!"}, status=status.HTTP_200_OK)


class PatientBeforeAfterImageView(APIView):
    permission_classes = ([IsAuthenticated])
    parser_classes = [MultiPartParser, FileUploadParser, FormParser]

    def get_patient_object(self, pk):
        try:
            return PatientDetails.objects.get(pk=pk)
        except PatientDetails.DoesNotExist:
            return status.HTTP_404_NOT_FOUND

    def get_image_object(self, pk):
        try:
            return PatientBeforeAfter.objects.get(pk=pk)
        except PatientBeforeAfter.DoesNotExist:
            return status.HTTP_404_NOT_FOUND

    def get(self, request, pk, *args, **kwargs):
        qs = PatientBeforeAfter.objects.filter(patient=pk)
        
        if len(qs) == 0:
            return Response({"message": "No images!"}, status=status.HTTP_204_NO_CONTENT)
        elif qs[0].patient.doctor.id == request.user.id:
            serializer = PatientBeforeAfterSerializer(qs, many=True)
            return Response({"resp": serializer.data})
        else:
            return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        

    def post(self, request, pk, *args, **kwargs):
        qs = self.get_patient_object(pk)
        print(qs)
        if qs != 404:
            dir_path = os.path.dirname(__file__)
            print(dir_path)
            img_path = dir_path + '/static/kayaApp/images'
            print(img_path)
            if qs.doctor.id == request.user.id:
                # image = dict((request.data).lists())['image']
                if len(request.query_params) == 0:
                    image = request.data['image']
                    image_name = os.path.splitext(image.name)

                else:
                    for i,j in request.query_params.items():
                        print(i,j)
                        qs2 = PatientImage.objects.get(pk=j)
                        image = qs2.image
                        image_name = image.name.split('/')[-1]
                        image_name = os.path.splitext(image_name)

                # print(image_name,"image_nameimage_name")
                # u_img = image[0]['name'].split('/')
                # img = random.choice(os.listdir(img_path))
                # print(img)
                # image_file = os.path.join(img_path, img)
                # print(image_file)
                # im1 = Image.open(image[0]['name'])
                im2 = Image.open(image)

                _,region = skin_prediction(im2)
                mask = region['masks']
                skin_prob = region['class_ids']

                after_image = skin_diagnosis.before_after(im2,mask,region)

                after_image = Image.fromarray(after_image)
                
                buffer1 = BytesIO()
                buffer2 = BytesIO()
                try:
                    im2.save(buffer1, format='JPEG', quality=85)
                    after_image.save(buffer2, format='JPEG', quality=85)
                except:
                    im2.save(buffer1, format='PNG', quality=85)
                    after_image.save(buffer2, format='PNG', quality=85)
                buffer1.seek(0)
                buffer2.seek(0)
                upload_pic= InMemoryUploadedFile(buffer1, 'ImageField',
                                    image.name,
                                    'image/jpeg',
                                    sys.getsizeof(buffer1), None)
                new_pic= InMemoryUploadedFile(buffer2, 'ImageField',
                                    image_name[0] + '_processed' + image_name[1],
                                    'image/jpeg',
                                    sys.getsizeof(buffer2), None)
                modified_data = before_after_image(upload_pic,new_pic)
                serializer = PatientBeforeAfterSerializer(data=modified_data)
                if serializer.is_valid():
                    serializer.save(patient=qs)
                    return Response(serializer.data, status=status.HTTP_200_OK)
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Patient does not exist!"}, status=status.HTTP_404_NOT_FOUND)        

    def patch(self, request, pk):
        qs = self.get_image_object(pk)
        if qs != 404:
            # dir_path = os.path.dirname(__file__)
            # print(dir_path)
            # img_path = dir_path + '/static/kayaApp/images'
            # print(img_path)
            if qs.patient.doctor.id == request.user.id:
                if len(request.query_params) == 0:
                    image = request.data['image']
                    image_name = os.path.splitext(image.name)

                else:
                    for i,j in request.query_params.items():
                        print(i,j)
                        qs2 = PatientImage.objects.get(pk=j)
                        image = qs2.image
                        image_name = image.name.split('/')[-1]
                        image_name = os.path.splitext(image_name)

                # print(image_name,"image_nameimage_name")
                # u_img = image[0]['name'].split('/')
                # img = random.choice(os.listdir(img_path))
                # print(img)
                # image_file = os.path.join(img_path, img)
                # print(image_file)
                # im1 = Image.open(image[0]['name'])
                im2 = Image.open(image)

                _,region = skin_prediction(im2)
                mask = region['masks']
                skin_prob = region['class_ids']

                after_image = skin_diagnosis.before_after(im2,mask,region)

                after_image = Image.fromarray(after_image)
                
                buffer1 = BytesIO()
                buffer2 = BytesIO()
                try:
                    im2.save(buffer1, format='JPEG', quality=85)
                    after_image.save(buffer2, format='JPEG', quality=85)
                except:
                    im2.save(buffer1, format='PNG', quality=85)
                    after_image.save(buffer2, format='PNG', quality=85)

                buffer1.seek(0)
                buffer2.seek(0)
                upload_pic= InMemoryUploadedFile(buffer1, 'ImageField',
                                    image.name,
                                    'image/jpeg',
                                    sys.getsizeof(buffer1), None)
                new_pic= InMemoryUploadedFile(buffer2, 'ImageField',
                                    image_name[0] + '_processed' + image_name[1],
                                    'image/jpeg',
                                    sys.getsizeof(buffer2), None)
                modified_data = before_after_image(upload_pic,new_pic)
                serializer = PatientBeforeAfterSerializer(qs,data=modified_data,partial = True)
                if serializer.is_valid():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_200_OK)
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Image does not exist!"}, status=status.HTTP_404_NOT_FOUND)

    def delete(self,request,pk):
        qs = self.get_image_object(pk)
        if qs != 404:
            if qs.patient.doctor.id == request.user.id:
                qs.delete()
                return Response({"message":"Record has been deleted!"},status = status.HTTP_200_OK)
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Image does not exist!"}, status=status.HTTP_404_NOT_FOUND)


class PatientMeasureImprovementView(APIView):
    permission_classes = ([IsAuthenticated])
    parser_classes = [MultiPartParser, FileUploadParser, FormParser]

    def get_patient_object(self, pk):
        try:
            return PatientDetails.objects.get(pk=pk)
        except PatientDetails.DoesNotExist:
            return status.HTTP_404_NOT_FOUND

    def get_image_object(self, pk):
        try:
            return PatientMeasureImprovement.objects.get(pk=pk)
        except PatientMeasureImprovement.DoesNotExist:
            return status.HTTP_404_NOT_FOUND

    def get(self, request, pk, *args, **kwargs):
        qs = PatientMeasureImprovement.objects.filter(patient=pk)
        print('here----',qs)

        if len(qs) == 0:
            return Response({"message": "No images!"}, status=status.HTTP_204_NO_CONTENT)
        elif qs[0].patient.doctor.id == request.user.id:
            serializer = PatientMeasureImprovementSerializer(qs,many=True)
            return Response({"resp": serializer.data})
        else:
            return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)

    def post(self, request, pk):
        qs = self.get_patient_object(pk)
        if qs != 404:
            if qs.doctor.id == request.user.id:
                if len(request.query_params) == 0:
                    images = dict((request.data).lists())['image']
                else:
                    images = []
                    for i,j in request.query_params.items():
                        print(i,j)
                        qs3 = PatientImage.objects.get(pk=j)
                        images.append(qs3.image)

                print(images)
                flag = 1
                arr = []
                remark = dict((request.data).lists())['remark']
                
                for img_name, remark in zip(images, remark):
                    u_img = img_name.name
                    print(u_img,"IMAGENAME")
                    im1 = Image.open(img_name)
                    buffer1 = BytesIO()
                    try:
                        im1.save(buffer1, format='JPEG', quality=85)
                    except:
                        im1.save(buffer1, format='PNG', quality=85)
                    buffer1.seek(0)
                    upload_pic= InMemoryUploadedFile(buffer1, 'ImageField',
                                    u_img,
                                    'image/jpeg',
                                    sys.getsizeof(buffer1), None)
                    #im2 = Image.open() for process image
                    if remark=="True":
                        qs2 = PatientMeasureImprovement.objects.filter(patient=qs.id).order_by('-id')
                    else:
                        qs2 = PatientMeasureImprovement.objects.filter(patient=qs.id).order_by('id')
                    # if (str(qs2[0].created_date) < str(datetime.datetime.now() + datetime.timedelta(days=30))):
                    print(qs2,"QAAAAAAAAAAAA")

                    if len(qs2)!=0:
                        if (str(qs2[0].created_date) < str(datetime.datetime.now())):
                            print('here')

                            comp_name = qs2[0].image

                            compared_img = Image.open(qs2[0].image)
                            buffer2 = BytesIO()
                            try:
                                compared_img.save(buffer2, format='JPEG', quality=85)
                            except:
                                compared_img.save(buffer2, format='PNG', quality=85)
                            buffer2.seek(0)
                            comp_pic= InMemoryUploadedFile(buffer2, 'ImageField',
                                    "comp.jpg",
                                    'image/jpeg',
                                    sys.getsizeof(buffer2), None)
                            

                            # image = Image.open(img_name)
                            ref_image,comp_region = skin_prediction(compared_img)
                            improved_img,region = skin_prediction(im1)
                            
                            comp_mask = comp_region['masks']
                            mask = region['masks']

                            ref_prob = comp_region['class_ids']
                            skin_prob = region['class_ids']

                            skin_score = 0.0
                            
                            if len(skin_prob)<len(ref_prob):
                                diff = len(ref_prob) - len(skin_prob)
                                if diff>5:
                                    skin_score = 10
                                elif diff==4:
                                    skin_score = 9
                                elif diff==3:
                                    skin_score = 8
                                elif diff==2:
                                    skin_score = 7
                                elif diff==1:
                                    skin_score = 6.5
                                elif diff==0:
                                    skin_score = 6
                                    
                            elif len(ref_prob)<len(skin_prob):
                                diff = len(skin_prob) - len(ref_prob)
                                if diff>5:
                                    skin_score = 5
                                elif diff==4:
                                    skin_score = 5.5
                                elif diff==3:
                                    skin_score = 6
                                elif diff==2:
                                    skin_score = 6.5
                                elif diff==1:
                                    skin_score = 6.75
                                elif diff==0:
                                    skin_score = 7
                            else:
                                skin_score = 7

                        modified_data = measure_improvement_multiple_files(upload_pic,remark,comp_pic, skin_score)
                        serializer = PatientMeasureImprovementSerializer(data=modified_data)
                        if serializer.is_valid():
                            data = serializer.save(patient=qs)
                            # print(serializer.data,"DATA")
                            arr.append(serializer.data)
                            # arr["id"] = data.id
                            # arr["image"] = data.image
                            # arr["skin_score"] =  data.skin_score
                            # arr["process_image"] = ""
                            # arr["created_date"] = data.created_date
                            # arr["updated_date"] = data.updated_date
                        else:
                            flag = 0

                    else:
                        skin_score = None
                        comp_pic = None
                        # process = None  #this will process output variable
                        modified_data = measure_improvement_multiple_files(upload_pic, remark,comp_pic, skin_score)
                        serializer = PatientMeasureImprovementSerializer(data=modified_data)
                        if serializer.is_valid():
                            data = serializer.save(patient=qs)
                            # print(serializer.data,"DATA")
                            arr.append(serializer.data)
                            # arr["id"] = data.id
                            # arr["image"] = data.image
                            # arr["skin_score"] =  data.skin_score
                            # arr["process_image"] = ""
                            # arr["created_date"] = data.created_date
                            # arr["updated_date"] = data.updated_date
                            
                        else:
                            flag = 0
                        
                if flag == 1:
                    print(arr,"ARRRR")
                    return Response({"resp":arr}, status=status.HTTP_201_CREATED)
                else:
                    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Patient does not exist!"}, status=status.HTTP_404_NOT_FOUND)

    def patch(self, request, pk):
        qs = self.get_image_object(pk)
        if qs != 404:
            if qs.patient.doctor.id == request.user.id:
                if len(request.query_params) == 0:
                    images = request.data['image']
                else:
                    for i,j in request.query_params.items():
                        print(i,j)
                        qs3 = PatientImage.objects.get(pk=j)
                        images = qs3.image

                print(images)
                flag = 1
                arr = []
                remark = request.data['remark']
            
                u_img = images.name
                print(u_img,"IMAGENAME")
                im1 = Image.open(images)
                buffer1 = BytesIO()
                try:
                    im1.save(buffer1, format='JPEG', quality=85)
                except:
                    im1.save(buffer1, format='PNG', quality=85)
                buffer1.seek(0)
                upload_pic= InMemoryUploadedFile(buffer1, 'ImageField',
                                u_img,
                                'image/jpeg',
                                sys.getsizeof(buffer1), None)
                #im2 = Image.open() for process image
                # print(type(remark),"REMARKSSSSSSS")
                if remark=="True":
                    qs2 = PatientMeasureImprovement.objects.filter(patient=qs.patient).order_by('-id')
                else:
                    qs2 = PatientMeasureImprovement.objects.filter(patient=qs.patient).order_by('id')
                # if (str(qs2[0].created_date) < str(datetime.datetime.now() + datetime.timedelta(days=30))):
                print(qs2[0].image,"QAAAAAAAAAAAA")
                # print(qs2[-1].image,"PREV")
                for i in qs2:
                    print(i.image,"PREVVVV")
                if len(qs2)!=0:
                    if (str(qs2[0].created_date) < str(datetime.datetime.now())):
                        print('here')

                        comp_name = qs2[0].image
                        print(qs2[0].image,"IMAGEEEEEEEE")

                        compared_img = Image.open(qs2[0].image)
                        buffer2 = BytesIO()
                        try:
                            compared_img.save(buffer2, format='JPEG', quality=85)
                        except:
                            compared_img.save(buffer2, format='PNG', quality=85)
                        buffer2.seek(0)
                        comp_pic= InMemoryUploadedFile(buffer2, 'ImageField',
                                "comp.jpg",
                                'image/jpeg',
                                sys.getsizeof(buffer2), None)
                        

                        # image = Image.open(img_name)
                        ref_image,comp_region = skin_prediction(compared_img)
                        improved_img,region = skin_prediction(im1)
                        
                        comp_mask = comp_region['masks']
                        mask = region['masks']

                        ref_prob = comp_region['class_ids']
                        skin_prob = region['class_ids']

                        skin_score = 0.0
                        
                        if len(skin_prob)<len(ref_prob):
                            diff = len(ref_prob) - len(skin_prob)
                            if diff>5:
                                skin_score = 10
                            elif diff==4:
                                skin_score = 9
                            elif diff==3:
                                skin_score = 8
                            elif diff==2:
                                skin_score = 7
                            elif diff==1:
                                skin_score = 6.5
                            elif diff==0:
                                skin_score = 6
                                
                        elif len(ref_prob)<len(skin_prob):
                            diff = len(skin_prob) - len(ref_prob)
                            if diff>5:
                                skin_score = 5
                            elif diff==4:
                                skin_score = 5.5
                            elif diff==3:
                                skin_score = 6
                            elif diff==2:
                                skin_score = 6.5
                            elif diff==1:
                                skin_score = 6.75
                            elif diff==0:
                                skin_score = 7
                        else:
                            skin_score = 7

                    modified_data = measure_improvement_multiple_files(upload_pic,remark,comp_pic, skin_score)
                    serializer = PatientMeasureImprovementSerializer(qs,data=modified_data,partial = True)
                    if serializer.is_valid():
                        data = serializer.save()
                        # print(serializer.data,"DATA")
                        arr.append(serializer.data)
                        # arr["id"] = data.id
                        # arr["image"] = data.image
                        # arr["skin_score"] =  data.skin_score
                        # arr["process_image"] = ""
                        # arr["created_date"] = data.created_date
                        # arr["updated_date"] = data.updated_date
                    else:
                        flag = 0

                else:
                    skin_score = None
                    comp_pic = None
                    # process = None  #this will process output variable
                    modified_data = measure_improvement_multiple_files(upload_pic, remark,comp_pic, skin_score)
                    serializer = PatientMeasureImprovementSerializer(qs,data=modified_data,partial = True)
                    if serializer.is_valid():
                        data = serializer.save()
                        # print(serializer.data,"DATA")
                        arr.append(serializer.data)
                        # arr["id"] = data.id
                        # arr["image"] = data.image
                        # arr["skin_score"] =  data.skin_score
                        # arr["process_image"] = ""
                        # arr["created_date"] = data.created_date
                        # arr["updated_date"] = data.updated_date
                        
                    else:
                        flag = 0
                        
                if flag == 1:
                    # print(arr,"ARRRR")
                    return Response({"resp":arr}, status=status.HTTP_201_CREATED)
                else:
                    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Image does not exist!"}, status=status.HTTP_404_NOT_FOUND)

    def delete(self, request, pk):
        qs = self.get_image_object(pk)
        if qs != 404:
            if qs.patient.doctor.id == request.user.id:
                qs.delete()
                return Response({"message":"Record has been deleted!"},status=status.HTTP_200_OK)
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Image does not exist!"}, status=status.HTTP_404_NOT_FOUND)


class ServiceView(APIView):
    permission_classes = ([IsAuthenticated])

    def get_patient_object(self, pk):
        try:
            return PatientDetails.objects.get(pk=pk)
        except PatientDetails.DoesNotExist:
            return status.HTTP_404_NOT_FOUND
    

    def get(self, request, pk):
        image_file = '/media/kayaApp/product/images/cream_tube.jpg'

        patient_query = self.get_patient_object(pk)
        try:
            qs1 = PatientForm2.objects.get(patient=pk)
        except PatientForm2.DoesNotExist:
            return Response(status = status.HTTP_204_NO_CONTENT)
        skin_type = qs1.skin_type
        print(skin_type)
        answer = qs1.answer
        print(answer)
        answer_type = qs1.answer_type
        print(answer_type)
        skin_type = skin_type.strip("][").split("', '")
        print(skin_type)
        answer = answer.strip("][").split("', '")
        print(answer)
        answer_type = answer_type.strip("][").split("', '")
        print(answer_type)
        concern_id_list = []
        ans = []
        for skin in skin_type:
            skin = str(skin.replace("'",""))
            for i in answer_type:  # till here
                print(skin, '---------------')
                i = str(i.replace("'",""))
                print(i, '---------------')
                qs2 = ConcernList.objects.filter(Q(concern__type=skin) & Q(concern__answer_type=i))
                if len(qs2) != 0:
                    for j in qs2:
                        concern_id_list.append(j.id)
                        ans.append(j.concern['answer'])
                else:
                    continue
        print(concern_id_list)
        service_list = []
        service_id_list = []
        for id in concern_id_list:
            qs3 = ServiceList.objects.get(concern=id)
            service_list.append(qs3.service)
            service_id_list.append(qs3.id)
        print(service_list)
        print(service_id_list)
        print(ans)
        li = []
        
        for services in range(len(service_list)):
            data={}
            data['concern'+str(services + 1)] = ans[services]
            print(ans[services])
            data['service'+str(services + 1)] = []
            for key, value in service_list[services].items():
                data['service'+str(services + 1)].append({"service_name":value})  #till here service list logic
            li.append(data)

        print("li-----------",li)
    
        age = patient_query.age     #from here product list logic
        num = 1
        for services, count in zip(service_id_list, range(len(li))):
            print(services)
            # data['product'+str(num)] = []
            li[count]['product'+str(num)] = []
            qs4 = ProductList.objects.get(service=services)
            # print(qs4.service_id)
            if skin_type == 'oily' or skin_type == 'sensitive':
                for key, value in qs4.product['age19'][0].items():
                    print(value)
                    li[count]['product'+str(num)].append(
                        {
                            "product_name":value,
                            "product_image":image_file
                        }
                    )
                    # data['product'+str(num)].append(
                    #     {
                    #         "product_name":value,
                    #         "product_image":image_file,
                    #         "concern":ans[num - 1]
                    #     }
                    # )

            else:
                print('here')
                if skin_type == 'dry':
                    if 20 <= services <= 21:
                        if 25 <= age < 35:
                            for key, value in qs4.product['age25'][0].items():
                                print(value)
                                li[count]['product'+str(num)].append(
                                    {
                                        "product_name":value,
                                        "product_image":image_file
                                    }
                                )
                                # data['product'+str(num)].append(
                                #     {
                                #         "product_name":value,
                                #         "product_image":image_file,
                                #         "concern":ans[num - 1]
                                #     }
                                # )

                        elif age >= 35:
                            for key, value in qs4.product['age35'][0].items():
                                print(value)
                                li[count]['product'+str(num)].append(
                                    {
                                        "product_name":value,
                                        "product_image":image_file
                                    }
                                )
                                # data['product'+str(num)].append(
                                #     {
                                #         "product_name":value,
                                #         "product_image":image_file,
                                #         "concern":ans[num - 1]
                                #     }
                                # )

                    elif 22 <= services <= 25:
                        if 25 <= age < 40:
                            for key, value in qs4.product['age25'][0].items():
                                print(value)
                                li[count]['product'+str(num)].append(
                                    {
                                        "product_name":value,
                                        "product_image":image_file
                                    }
                                )
                                # data['product'+str(num)].append(
                                #     {
                                #         "product_name":value,
                                #         "product_image":image_file,
                                #         "concern":ans[num - 1]
                                #     }
                                # )

                        elif age >= 40:
                            for key, value in qs4.product['age40'][0].items():
                                print(value)
                                li[count]['product'+str(num)].append(
                                    {
                                        "product_name":value,
                                        "product_image":image_file
                                    }
                                )
                                # data['product'+str(num)].append(
                                #     {
                                #         "product_name":value,
                                #         "product_image":image_file,
                                #         "concern":ans[num - 1]
                                #     }
                                # )
                    else:
                        pass
                elif 19 <= age < 35:                    
                    for key, value in qs4.product['age19'][0].items():
                        print(value)
                        li[count]['product'+str(num)].append(
                            {
                                "product_name":value,
                                "product_image":image_file
                            }
                        )
                        # data['product'+str(num)].append(
                        #     {
                        #         "product_name":value,
                        #         "product_image":image_file,
                        #         "concern":ans[num - 1]
                        #     }
                        # )
                elif age >= 35:
                    for key, value in qs4.product['age35'][0].items():
                        print(value)
                        li[count]['product'+str(num)].append(
                            {
                                "product_name":value,
                                "product_image":image_file
                            }
                        )
                        # data['product'+str(num)].append(
                        #     {
                        #         "product_name":value,
                        #         "product_image":image_file,
                        #         "concern":ans[num - 1]
                        #     }
                        # )
            num +=1

        return Response({"resp": li})

    def post(self, request, pk):
        
        qs = self.get_patient_object(pk)
        if qs != 404:
            if qs.doctor.id == request.user.id:
                serializer = PatientServiceProductSerializer(data=request.data)
                if serializer.is_valid():
                    serializer.save(patient=qs, service=request.data['service'], product=request.data['product'])
                    return Response(serializer.data, status=status.HTTP_200_OK)
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "Patient does not exist!"}, status=status.HTTP_404_NOT_FOUND)

    def patch(self, request, pk):
        
        qs = PatientServiceProduct.objects.get(patient=pk)
        if qs != 404:
            if qs.patient.doctor.id == request.user.id:
                serializer = PatientServiceProductSerializer(qs, data=request.data, partial=True)
                if serializer.is_valid():
                    serializer.save(service=request.data['service'], product=request.data['product'])
                    return Response(serializer.data, status=status.HTTP_200_OK)
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED) 
        else:
            return Response({"error": "Patient does not exist!"}, status=status.HTTP_404_NOT_FOUND)
        

class CheckForm1(APIView):
    permission_classes = ([IsAuthenticated])

    def get_form1_object(self, pk):
        try:
            return PatientForm1.objects.get(patient=pk)
        except PatientForm1.DoesNotExist:
            return status.HTTP_404_NOT_FOUND

    def get(self, request, pk):
        qs = self.get_form1_object(pk)
        # print(qs.patient.doctor.id)
        if qs != 404:
            if qs.patient.doctor.id == request.user.id:
                return Response(status=status.HTTP_200_OK)
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response(status=status.HTTP_404_NOT_FOUND)


class CheckForm2(APIView):
    permission_classes = ([IsAuthenticated])

    def get_form2_object(self, pk):
        try:
            return PatientForm2.objects.get(patient=pk)
        except PatientForm2.DoesNotExist:
            return status.HTTP_404_NOT_FOUND

    def get(self, request, pk):
        qs = self.get_form2_object(pk)
        # print(qs.patient.doctor.id)
        if qs != 404:
            if qs.patient.doctor.id == request.user.id:
                return Response(status=status.HTTP_200_OK)
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response(status=status.HTTP_404_NOT_FOUND)


class CheckData(APIView):
    permission_classes = ([IsAuthenticated])

    def get_service_product_object(self, pk):
        try:
            return PatientServiceProduct.objects.get(patient=pk)
        except PatientServiceProduct.DoesNotExist:
            return status.HTTP_404_NOT_FOUND

    def get(self, request, pk):
        qs = self.get_service_product_object(pk)
        # print(qs.patient.doctor.id)
        if qs != 404:
            if qs.patient.doctor.id == request.user.id:
                return Response({"resp": True})
            else:
                return Response({"error": "Unauthorized!"}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"resp": False})


# class ProductView(APIView):

#     def get(self, request, pk):
#         qs1 = PatientForm2.objects.get(patient=pk)
#         skin_type = qs1.skin_type
#         print(skin_type)
#         answer = qs1.answer
#         # print(answer)
#         answer = answer.strip("][").split("', '")
#         # print(answer)
#         concern_id_list = []
#         for i in answer:
#             i = str(i.replace("'",""))
#             # print(i)
#             qs2 = ConcernList.objects.filter(Q(concern__type=skin_type) & Q(concern__answer=i))
#             if len(qs2) != 0:
#                 for j in qs2:
#                     concern_id_list.append(j.id)
#             else:
#                 continue
#         print(concern_id_list)
#         service_id_list = []
#         for id in concern_id_list:
#             qs3 = ServiceList.objects.get(concern=id)
#             service_id_list.append(qs3.id)
#         print(service_id_list)
        
#         qs3 = PatientDetails.objects.get(pk=pk)
#         age = qs3.age
#         data = {}
#         num = 1
#         for id in service_id_list:
#             data['product'+str(num)] = []
#             print(id)
#             qs4 = ProductList.objects.get(service=id)
#             if skin_type == 'oily' or skin_type == 'sensitive':
#                 pass
#             else:
#                 print('here')
#                 if skin_type == 'dry':
#                     if 20 <= id <= 21:
#                         if 25 <= age < 35:
#                             pass
#                         elif age >= 35:
#                             pass
#                     elif 22 <= id <= 25:
#                         if 25 <= age < 45:
#                             pass
#                         elif age >= 45:
#                             pass
#                 elif 19 <= age < 35:                    
#                     for key, value in qs4.product['age19'][0].items():
#                         print(value)
#                         data['product'+str(num)].append(value)
#             # print(data)
#             num += 1
                    

#             # print(qs4)
         


#         return Response({"Resp": [data]})



# @api_view(['GET'])
# @permission_classes([AllowAny])
# def testing_from(APIView):
#     print(os.path)
#     data =[ {
#         "id": 2,
#         "question_type": {
#             "type": "general",
#             "question": "Are you pregnant:",
#             "question_options": {
#                 "option": [
#                     {
#                         "option1": {
#                             "text":[
#                                 {"data1": "Yes"},
#                                 {"data2": "None"}
#                             ]
#                         }
#                     }
#                 ]
#             }
#         }
#     }]
#     return Response({"resp": data})



# def tooth(request):
#     mask_path = None
#     if request.method=="POST":
#         img= request.FILES['tooth_img']
#         img_name = secure_filename(img.name)
#         print("\n\n\n\nProcessing...")
#         with open(os.path.join(ROOT_DIR,'tooth_demo/decay/static/',img_name), 'wb+') as destination:
#             for chunk in img.chunks():
#                 destination.write(chunk)

#         print("\n\nAnalysing X-Ray...")
#         img_path = '../static/{}'.format(img_name)
#         mask_path = prediction(img_name)
#         return render(request,"index.html",{'mask_path':mask_path})
#     return render(request,"index.html")

# def prediction(img_name):
#     # Load weights
#     with graph.as_default():
#         image = cv2.imread(os.path.join(ROOT_DIR,'tooth_demo/decay/static/',img_name))
#         results = model.detect(image, verbose=0)

#     r = results[0]
#     Classes = ['BG', 'decay']
#     masked_img = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                                 Classes, r['scores'],
#                                 title="Predictions")

#     mask_path = '../static/mask{}'.format(img_name)
#     cv2.imwrite(os.path.join(ROOT_DIR,'tooth_demo/decay/static/mask{}'.format(img_name)),masked_img)
#     print("\n\nProcessing Complete")
#     return mask_path