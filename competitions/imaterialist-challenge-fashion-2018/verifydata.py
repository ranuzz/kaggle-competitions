import os
import json
import requests
from multiprocessing import Process
import time
import matplotlib.image as mpimg
from PIL import Image

from utils.AppsHelper import KaggleApp
from fetch.single import ConcurrentDownload, DownloadFile
from appsettings import COMPETITION_NAME,\
    ROOT_DATA_DIR, COMPETITION_DATA,\
    COMPETITION_FILE_LIST, TEST_DATA,\
    TRAIN_DATA, VAL_DATA, IMAGE_PATH,\
    LABEL_FILE

"""
Train : 15321, 15926
"""
START_IMAGE_ID = 928622

def readKaggleFiles(cdata_location):
    # dataset
    test_dataset = "test.json"
    train_dataset = "train.json"
    validation_dataset = "validation.json"

    with open(os.path.join(cdata_location, train_dataset)) as fp:
        train_json = json.loads(fp.read())
    with open(os.path.join(cdata_location, test_dataset)) as fp:
        test_json = json.loads(fp.read())
    with open(os.path.join(cdata_location, validation_dataset)) as fp:
        validation_json = json.loads(fp.read())

    return test_json, train_json, validation_json

def VerifyFetchImageFiles(applog, json_data, download_path):
    for image_info in json_data['images']:
        url = image_info['url']
        imageId = image_info['imageId']
        if int(imageId) < START_IMAGE_ID:
            continue
        if not os.path.isfile(os.path.join(download_path, str(imageId))):
            applog.info("Downloading fresh " + url)
            DownloadFile(url, download_path, str(imageId))
        else:
            try:
                img = Image.open(os.path.join(download_path, str(imageId)))
                img = img.resize((512, 512))
                applog.info("File OK : [Image ID : " + str(imageId) + " ] : "+ url)
            except FileNotFoundError:
                try:
                    os.remove(os.path.join(download_path, str(imageId)))
                except:
                    pass
                applog.error("FileNotFound : Retry : " + url)
                DownloadFile(url, download_path, str(imageId))
            except OSError:
                try:
                    os.remove(os.path.join(download_path, str(imageId)))
                except:
                    pass
                applog.error("OSError : Retyr : " + url)
                DownloadFile(url, download_path, str(imageId))

if __name__ == '__main__':

    appname = COMPETITION_NAME + "_verifydata"
    print("starting verification for : " + COMPETITION_NAME)
    application = KaggleApp(appname, COMPETITION_NAME)
    applogger = application.getLogger()

    applogger.info("instance ID : %s", application.instanceid)
    applogger.info("instance name : %s", application.appname)

    if not application.check_data_dir():
        applogger.error("not able to create/locate data directory")
        application.quick_exit()

    # main logic
    applogger.info('verifying kaggle data')
    root_datadir_path = os.path.join(application.data_dir,
                                            ROOT_DATA_DIR)
    comp_datadir_path = os.path.join(application.data_dir,
                                            ROOT_DATA_DIR,
                                            COMPETITION_DATA)
    applogger.info('reading competition data from json files')
    try:
        test_json, train_json, validation_json = readKaggleFiles(comp_datadir_path)
    except Exception as e:
        print("Can not find Kaggle data: Run preprocessor before verification")
        print(e)

    """
    test_data_path = os.path.join(root_datadir_path, TEST_DATA, IMAGE_PATH)
    if application.check_any_dir(test_data_path):
        applogger.info('verifying images from test dataset')
        VerifyFetchImageFiles(applogger, test_json, test_data_path)
    """

    """
    val_data_path = os.path.join(root_datadir_path, VAL_DATA, IMAGE_PATH)
    if application.check_any_dir(val_data_path):
        applogger.info('verifying images from validation dataset')
        VerifyFetchImageFiles(applogger, validation_json, val_data_path)
    """

    train_data_path = os.path.join(root_datadir_path, TRAIN_DATA, IMAGE_PATH)
    if application.check_any_dir(train_data_path):
        applogger.info('verifying images from training dataset')
        VerifyFetchImageFiles(applogger, train_json, train_data_path)