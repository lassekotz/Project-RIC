from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def drive_upload(file_list):
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)

    for upload_file in file_list:
        gfile = drive.CreateFile({'parents': [{'id': '1pzschX3uMbxU0lB5WZ6IlEEeAUE8MZ-t'}]})
        # Read file and set it as the content of this instance.
        gfile.SetContentFile(upload_file)
        gfile.Upload() # Upload the file.