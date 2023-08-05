import os
import cv2
import requests


def save_image(image):
    img_count = 0
    img_name = f'image_{img_count}.png'
    path_img = 'C:/Rafif/SKRIPSI/Proyek Skripsi - Pycharm/img/'
    img_count += 1
    cv2.imwrite(os.path.join(path_img, img_name), image)
    files = {'photo': open(path_img + img_name, 'rb')}
    return files


def send_msg(caption, files):
    token = "5870827651:AAH3AjqVoCO6zmKraw6a8kOlud8HCcCDLvc"
    chat_id = "1841767294"
    url = "https://api.telegram.org/bot"
    captions = {}
    file_path = 'caption.txt'
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            captions[i + 1] = line.strip()
    caption = captions[caption]
    url_req = requests.post(url + token + "/sendPhoto" + "?chat_id=" + chat_id + "&caption=" + caption + '',
                            files=files)
    return url_req
