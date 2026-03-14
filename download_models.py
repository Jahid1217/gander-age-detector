import urllib.request
import os
import sys

# these URLs are standard for this tutorial.
urls = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "age_deploy.prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt",
    "age_net.caffemodel": "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
    "gender_deploy.prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt",
    "gender_net.caffemodel": "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel"
}

for name, url in urls.items():
    if not os.path.exists(name):
        print(f"Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, name)
            print(f"Downloaded {name} successfully.")
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            sys.exit(1)
    else:
        print(f"{name} already exists.")
print("All files downloaded.")
