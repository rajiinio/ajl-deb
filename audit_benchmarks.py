#!/usr/bin/env python
import requests
import os
import json
import boto3
import argparse
import base64
import csv
import subprocess
import time
from watson_developer_cloud import VisualRecognitionV3, WatsonApiException

K_API_ID = os.environ['K_API_ID']
K_API_KEY = os.environ['K_API_KEY']
I_API_KEY = os.environ['I_API_KEY']
M_SUB_KEY = os.environ['M_SUB_KEY']
F_API_KEY = os.environ['F_API_KEY']
F_API_SEC = os.environ['F_API_SEC']


class AuditBenchmark:
    def __init__(self,
                 k_api_id=K_API_ID,
                 k_api_key=K_API_KEY,
                 i_api_key=I_API_KEY,
                 m_sub_key=M_SUB_KEY,
                 f_api_key=F_API_KEY,
                 f_api_secret=F_API_SEC):

        self.k_api_id = k_api_id
        self.k_api_key = k_api_key

        self.i_api_key = i_api_key
        self.visual_recognition = VisualRecognitionV3('2018-03-19', iam_api_key=self.i_api_key)

        self.m_subscription_key = m_sub_key
        assert self.m_subscription_key

        self.f_api_key = f_api_key
        self.f_api_secret = f_api_secret

        self.cli = boto3.client('rekognition', 'us-east-1')

    def _image_to_bytes(self, filename, b64_format=False):
        with open(filename, "rb") as imageFile:
            if not b64_format:
                return imageFile.read()
            else:
                return str(base64.b64encode(imageFile.read()))[2:-1]

    def a_detect_faces(self, input):
        img_bytes = self._image_to_bytes(input)
        response = self.cli.detect_faces(
            Image={
            "Bytes": img_bytes,
            },
            Attributes=['ALL']
        )

        try:
            #confidence = response['FaceDetails'][0]['Gender']['Confidence']
            gender = response['FaceDetails'][0]['Gender']['Value'][0].lower()
        except:
            print('bad_request')
            print(response)
            gender = ['n/a']
            #confidence = 0

        return gender #confidence

    def k_detect_faces(self, input):
        img_bytes = self._image_to_bytes(input, True)
        # put your keys in the header
        headers = {
            "app_id": self.k_api_id,
            "app_key": self.k_api_key
        }
        payload = '{"image":"%s"}' %img_bytes
        url = "http://api.kairos.com/detect"
        count = 0
        while count <= 5:
            r = requests.post(url, data=payload, headers=headers)
            try:
                # make request
                res = json.loads(r.content)
                gender = res['images'][0]['faces'][0]['attributes']['gender']['type']
                break
            except:
                print('bad_request %d' %count)
                print(r.content)
                print(r)
                gender = 'n/a'
                time.sleep(60)
                count += 1


        return gender.lower()

    def m_detect_faces(self, input):
        # You must use the same region in your REST call as you used to get your
        # subscription keys. For example, if you got your subscription keys from
        # westus, replace "westcentralus" in the URI below with "westus".
        #
        # Free trial subscription keys are generated in the westcentralus region.
        # If you use a free trial subscription key, you shouldn't need to change
        # this region.
        face_api_url = 'https://eastus.api.cognitive.microsoft.com/face/v1.0/detect'
        img_bytes = self._image_to_bytes(input)
        headers = {'Ocp-Apim-Subscription-Key': self.m_subscription_key,
               "Content-Type": "application/octet-stream"}
        params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,' +
                                'emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
        }
        count = 0
        while count <= 5:

            try:
                response = requests.post(face_api_url, params=params, headers=headers, data=img_bytes)
                face = response.json()
                gender = face[0]['faceAttributes']['gender'][0]
                break
            except:
                print('bad_request_m %d' %count)
                gender = 'n/a'
                time.sleep(60)
                count += 1

        return gender



    def i_detect_faces(self, input):
        img_bytes = self._image_to_bytes(input)
        try:
            face_result = self.visual_recognition.detect_faces(images_file=img_bytes)
            gender = face_result['images'][0]['faces'][0]['gender']['gender'][0].lower()
        except:
            gender = 'n/a'

        return gender

    def f_detect_faces(self, input):
        cmd = 'curl -X POST https://api-us.faceplusplus.com/facepp/v3/detect -F api_key=%s -F api_secret=%s -F image_file=@%s -F return_landmark=1 -F return_attributes=gender,age' %(self.f_api_key, self.f_api_secret, input)
        count = 0
        while count <= 30:
            try:
                response = subprocess.run(cmd.split(" "), stdout=subprocess.PIPE)
                gender = json.loads(response.stdout)['faces'][0]['attributes']['gender']['value'][0].lower()
                break
            except:
                print('bad_request %d' % count)
                print(response.stdout)
                gender = 'n/a'
                time.sleep(10)
                count += 1

        return gender

    def get_results(self, input_file, base_dir, filter):
        filter = filter.split(',')
        func_dict = {
            "A": ["Amazon", self.a_detect_faces],
            "K": ["Kairos", self.k_detect_faces],
            "M": ["Microsoft", self.m_detect_faces],
            "F": ["Face++", self.f_detect_faces],
            "I": ["IBM", self.i_detect_faces]
        }
        dict_list = [] # TO DO: use pandas dataframe instead of dict list
        reader = csv.DictReader(open(input_file, 'r'))
        for line in reader:

            #TO DO: Generalize to other target outputs
            gender = line['Gender'][0].lower()
            #TO DO: handle url input
            filename = base_dir + line['filename']

            # TO DO: cycle more effeciently through api companies
            for item in filter:
                print(item)
                predicted_gen = func_dict[item][1](filename)
                results_title = func_dict[item][0]+' Results'
                class_acc_title = 'Classifier ' + func_dict[item][0]
                line[results_title] = predicted_gen
                if predicted_gen == 'n/a':
                    line[class_acc_title] = 'NA'
                elif predicted_gen == gender:
                    line[class_acc_title] = 'TRUE'
                else:
                    line[class_acc_title] = 'FALSE'

            print(line)
            dict_list.append(line)

        return dict_list

    def write_results(self, dict_list, output_file):
        with open(output_file, 'w') as csvfile:
            fieldnames = dict_list[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in dict_list:
                writer.writerow(row)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_file', '-i', required=True, help='input csv file with benchmark data')
    parser.add_argument('--output_csv_file', '-o', default=None, help='output csv file with audit results for api')
    parser.add_argument('--url_input', '-u', action='store_true', help='Flag if url input instead of local file input')
    parser.add_argument('--base_dir', '-b', default=None, help='base directory for input files')
    parser.add_argument('--filter', '-f', default='A,K,M,F,I', help='Indicate company APIs to be audited, represented by first letter of company names, seperated by commas. '
                                                                    'Default is "A,K,M,F,I" for all available APIs - Amazon, Kairos, Microsoft, Face++, and IBM. '
                                                                    'Any desired subset of this list is allowed, excluding an empty list.   ')

    args = parser.parse_args()

    audit = AuditBenchmark()
    # regardless, you want to print the numeric results
    data_dict = audit.get_results(args.input_csv_file, args.base_dir, args.filter)

    if args.output_csv_file:
        audit.write_results(data_dict, args.output_csv_file)









