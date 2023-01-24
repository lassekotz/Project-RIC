from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def drive_upload(path = r"/home/ric/Documents/projectRIC/actuation/Images"):
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    gauth.LocalWebserverAuth()

    for x in os.listdir(path):
        f = drive.CreateFile({'title': x})
        f.SetContentFile(os.path.join(path, x))
        f.Upload()

        f = None