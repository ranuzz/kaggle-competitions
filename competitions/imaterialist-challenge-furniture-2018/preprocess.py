import os
import json
import requests
from multiprocessing import Process
import time
import sys

from utils.AppsHelper import KaggleApp
from fetch.single import ConcurrentDownload, DownloadFile
from appsettings import COMPETITION_NAME, \
    ROOT_DATA_DIR, COMPETITION_DATA, \
    COMPETITION_FILE_LIST, TEST_DATA, \
    TRAIN_DATA, VAL_DATA, IMAGE_PATH, \
    LABEL_FILE


def VerifyKaggleData(app):
    assert isinstance(app, KaggleApp)
    if not app.check_data_dir(ROOT_DATA_DIR):
        return False
    if not app.check_data_dir(os.path.join(ROOT_DATA_DIR, COMPETITION_DATA)):
        return False
    comp_datadir_path = os.path.join(app.data_dir,
                                     ROOT_DATA_DIR,
                                     COMPETITION_DATA)
    flist_df = app.getCompetitionFileList(os.path.join(comp_datadir_path, COMPETITION_FILE_LIST))
    for name in flist_df['name']:
        if not os.path.isfile(os.path.join(comp_datadir_path, name)):
            app.downloadCompetitionFiles(comp_datadir_path, name)
            if not os.path.isfile(os.path.join(comp_datadir_path, name)):
                return False
        app.extractCompetitionFiles(comp_datadir_path, name)
    return True


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


def VerifyFetchImageFiles(json_data, download_path):
    download_threads = []
    for image_info in json_data['images']:
        url = image_info['url'][0]
        imageId = image_info['image_id']
        # print(url, imageId, download_path)
        if not os.path.isfile(os.path.join(download_path, str(imageId))):
            (url, download_path, str(imageId))
        """
        p = ConcurrentDownload(url, download_path, str(imageId))
        if p is not None:
            download_threads.append(p)
        WaitProcs(download_threads)
        """


def WaitProcs(procs, timeout=0):
    while len(procs) != 0:
        print("current active procs ", len(procs))
        rmproc = []
        for p in procs:
            if not p.is_alive():
                rmproc.append(p)
        for rmp in rmproc:
            procs.remove(rmp)
        time.sleep(900)


def FastDownload(json_data, download_path, start, end):
    #s = requests.Session()
    lcount = 0
    for image_info in json_data['images'][start:end]:
        url = image_info['url'][0]
        imageId = image_info['image_id']
        if os.path.isfile(os.path.join(download_path, str(imageId))):
            continue
        try:
            #r = s.get(url)
            DownloadFile(url, download_path, str(imageId))
            #with open(os.path.join(download_path, str(imageId)), 'wb') as fp:
            #    fp.write(r.content)
        except Exception as e:
            print(e)
            print("cannot download : " + url)
    #s.close()


def FasterDownload(json_data, download_path, procs, s, e, q, fs):
    start = s
    end = e
    fullstop = fs
    quantom = q
    proc_list = []
    for i in range(procs):
        p = Process(target=FastDownload, args=(json_data, download_path, start, end))
        print("starting procs for ", start, end)
        p.start()
        proc_list.append(p)
        start = end
        end = end + quantom
        if end > fullstop:
            end = fullstop - 1
    WaitProcs(proc_list)


def VerifyCreateLabels(json_data, labelfile):
    if not os.path.isfile(labelfile):
        fp = open(labelfile, "w")
        fp.write("id,predicted\n")
        for label_info in json_data['annotations']:
            fp.write(str(label_info['image_id']) + ",")
            fp.write(str(label_info['label_id']))
            fp.write("\n")


if __name__ == '__main__':

    appname = COMPETITION_NAME + "_preprocessing"
    print("starting preprocessing for : " + COMPETITION_NAME)
    application = KaggleApp(appname, COMPETITION_NAME)
    applogger = application.getLogger()

    applogger.info("instance ID : %s", application.instanceid)
    applogger.info("instance name : %s", application.appname)

    if not application.check_data_dir():
        applogger.error("not able to create/locate data directory")
        application.quick_exit()

    # main logic
    applogger.info('verifying kaggle data')
    if not VerifyKaggleData(application):
        applogger.error('competition data is not available')
        applogger.error('Please download manually')

    applogger.info('creating essential directories')
    root_datadir_path = os.path.join(application.data_dir,
                                     ROOT_DATA_DIR)
    comp_datadir_path = os.path.join(application.data_dir,
                                     ROOT_DATA_DIR,
                                     COMPETITION_DATA)

    applogger.info('reading competition data from json files')
    test_json, train_json, validation_json = readKaggleFiles(comp_datadir_path)

    """
    applogger.info('fetching images from test dataset')
    test_data_path = os.path.join(root_datadir_path, TEST_DATA, IMAGE_PATH)
    if application.check_any_dir(test_data_path):
        VerifyFetchImageFiles(test_json, test_data_path)

    applogger.info('fetching images from train dataset')
    train_data_path = os.path.join(root_datadir_path, TRAIN_DATA, IMAGE_PATH)
    if application.check_any_dir(train_data_path):
        VerifyFetchImageFiles(train_json, train_data_path)

    applogger.info('fetching images from validation dataset')
    val_data_path = os.path.join(root_datadir_path, VAL_DATA, IMAGE_PATH)
    if application.check_any_dir(val_data_path):
        VerifyFetchImageFiles(validation_json, val_data_path)
    """
    train_data_path = os.path.join(root_datadir_path, TRAIN_DATA, IMAGE_PATH)
    test_data_path = os.path.join(root_datadir_path, TEST_DATA, IMAGE_PATH)
    val_data_path = os.path.join(root_datadir_path, VAL_DATA, IMAGE_PATH)
    # FastDownload(train_json, train_data_path)

    try:
        os.makedirs(train_data_path)
        os.makedirs(test_data_path)
        os.makedirs(val_data_path)
    except:
        pass

    if not os.path.isdir(train_data_path):
        print(train_data_path + " : does not exists")
        sys.exist()
    if not os.path.isdir(test_data_path):
        print(test_data_path + " : does not exists")
        sys.exist()
    if not os.path.isdir(val_data_path):
        print(val_data_path + " : does not exists")
        sys.exist()

    applogger.info('fetching images from train dataset')
    nprocs = 15
    fs = len(train_json['images'])
    e = (fs // nprocs) + 1000
    FasterDownload(train_json, train_data_path, nprocs, 0, e, e, fs)

    # exit after doing proper cleanup
    """
        applogger.info('creating label files')
    train_label_file = os.path.join(root_datadir_path, TRAIN_DATA, LABEL_FILE)
    VerifyCreateLabels(train_json, train_label_file)
    val_label_file = os.path.join(root_datadir_path, VAL_DATA, LABEL_FILE)
    VerifyCreateLabels(validation_json, val_label_file)

    applogger.info('fetching images from val dataset')
    nprocs = 4
    fs = len(validation_json['images'])
    e = (fs // nprocs) + 1000
    FasterDownload(validation_json, val_data_path, nprocs, 0, e, e, fs)

    applogger.info('fetching images from test dataset')
    nprocs = 4
    fs = len(test_json['images'])
    e = (fs // nprocs) + 1000
    FasterDownload(test_json, test_data_path, nprocs, 0, e, e, fs)
    """