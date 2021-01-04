import numpy as np
import base64
import sys
import os
from os.path import join, splitext
import concurrent.futures
import time
import json
import requests


def base64_encode_image(a):
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode("utf-8")


def base64_decode_image(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodebytes(a), dtype=dtype)
    # a = a.reshape(shape)

    # return the decoded image
    return a


def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def post_img(img_path, url):
    img_b64 = get_base64_encoded_image(img_path)
    r = requests.post(url, data={'img_name': os.path.basename(img_path), 'img_b64': img_b64})
    res = r.text
    res_data = json.loads(res)
    return res_data

def send_invoice_images_to_post(img_dir, url):
    for filename in os.listdir(img_dir):
        if not splitext(filename)[-1] == '.jpg':
            print(filename)
            continue
        post_img(join(img_dir, filename), url)


def concurrent_post(img_dir, url):
    succecc_times = 0
    faild_times = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Start the load operations and mark each future with its URL
        file_names = [filename for filename in os.listdir(img_dir) if splitext(filename)[-1] == '.jpg']
        future_to_filename = {executor.submit(post_img, join(img_dir, filename), url): filename for filename in file_names}
        for future in concurrent.futures.as_completed(future_to_filename):
            filename = future_to_filename[future]
            try:
                data = future.result()
                if data is None:
                    faild_times += 1
                else:
                    succecc_times += 1
            except Exception as exc:
                print('%r generated an exception: %s' % (filename, exc))
            else:
                print('filename {}, data {}'.format(filename, data))
    print('succecc_times {}, faild_times {}'.format(succecc_times, faild_times))


if __name__ == '__main__':
    api_url = "http://127.0.0.1:8000/api/cache/setimage"
    img_dir = r'E:\python_workspace\chineseocr\data-for-test\vat_invoice_files'
    t1 = time.time()
    post_times = 0
    for i in range(10):
        # send_invoice_images_to_post(img_dir, api_url)
        concurrent_post(img_dir, api_url)
        post_times += len([filename for filename in os.listdir(img_dir) if splitext(filename)[-1] == '.jpg'])
    t2 = time.time()
    print('concurrent_post {} post, use {} seconds'.format(post_times, t2 - t1))
