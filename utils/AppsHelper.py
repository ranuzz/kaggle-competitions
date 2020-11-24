import uuid
import os
import logging
import sys
from subprocess import run, PIPE
from io import StringIO
import pandas as pd
import zipfile

from settings import LOG_DIR, DATA_DIR

class KaggleApp:

    def __init__(self, name, cname):
        self.appname = name
        self.competition_name = cname
        self.instanceid = str(uuid.uuid1())
        self.logger = None
        self.data_dir = DATA_DIR

    def logger_init(self):
        if self.logger is not None:
            return
        self.check_log_dir()
        logging.basicConfig(filename=os.path.join(LOG_DIR, str(self.instanceid) + ".log"),
                            filemode='a',
                            level=logging.NOTSET,
                            format='%(asctime)s : %(levelname)s : %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')
        self.logger = logging.getLogger(self.appname)

    def check_log_dir(self):
        if not os.path.isdir(LOG_DIR):
            try:
                os.makedirs(LOG_DIR, exist_ok=True)
            except Exception as e:
                print("not able to create log dir : ", LOG_DIR)
                print(e)
                self.quick_exit()

    def check_data_dir(self, rel_path=""):
        if not os.path.isdir(self.data_dir):
            try:
                os.makedirs(self.data_dir, exist_ok=True)
            except Exception as e:
                print("not able to create log dir : ", self.data_dir)
                print(e)
                return False
        if rel_path != "":
            if not os.path.isdir(os.path.join(self.data_dir, rel_path)):
                try:
                    os.makedirs(os.path.join(self.data_dir, rel_path), exist_ok=True)
                except Exception as e:
                    print("not able to create log dir : ", os.path.join(self.data_dir, rel_path))
                    print(e)
                    return False
        return True

    def check_any_dir(self, path):
        if not os.path.isdir(path):
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                print("not able to create dir : ", path)
                print(e)
                return False
        return True

    def getLogger(self):
        if self.logger is None:
            self.logger_init()
        return self.logger

    def getCompetitionFileList(self, path):
        if not path:
            return None
        if not os.path.isfile(path):
            output = None
            try:
                output = run(["kaggle", "competitions", "files", "-c", self.competition_name, "-v", "-q"], stdout=PIPE)
            except Exception as e:
                print(e)
                return None
            flist_csv = output.stdout.decode("utf-8")
            flist_df = pd.read_csv(StringIO(flist_csv))
            flist_df.to_csv(path)
            return flist_df
        else:
            flist_df = pd.read_csv(path)
            return flist_df

    def downloadCompetitionFiles(self, location, filename):
        # kaggle competitions download -c favorita-grocery-sales-forecasting -f test.csv.7z
        if os.path.isfile(filename):
            os.remove(filename)
        try:
            output = run(["kaggle", "competitions", "download", "-c",
                          self.competition_name, "-f", filename,
                          "-p", location])
        except Exception as e:
            print(e)

    def extractCompetitionFiles(self, location, filename):
        if zipfile.is_zipfile(os.path.join(location, filename)):
            zp = zipfile.ZipFile(os.path.join(location, filename))
            member_list = zp.namelist()
            for member in member_list:
                if not os.path.isfile(os.path.join(location, member)):
                    zp.extract(member, path=location)

    def quick_exit(self):
        sys.exit()