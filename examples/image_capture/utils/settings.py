import os

process_stat={}
def get_frame_save_dir():
    home = os.path.expanduser("~")
    frame_dir = os.path.join(home, '.streamsave/')
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    return frame_dir
