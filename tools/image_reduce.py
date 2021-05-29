import os
from io import BytesIO
from os import listdir
from os.path import join
from pathlib import PurePosixPath

from PIL import Image, ExifTags


def reduce_pil_image(img, max_size=1080):
    """
    压缩 PIL image对象
    :param max_size:
    :param img: PIL image对象
    :return: PIL image对象
    """
    try:
        # 获取图片拍摄角度,再处理是保持角度
        orientation = 0
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img.getexif().items())
        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except KeyError as e:
        print('KeyError, the key is {}'.format(e))
    except Exception as e:
        print('unkonwn Exception is {}'.format(e))
    xsize, ysize = img.size
    if xsize > max_size:
        ysize = int(max_size / xsize * ysize)
        xsize = max_size
    elif ysize > max_size:
        xsize = int(max_size / ysize * xsize)
        ysize = max_size
    img = img.resize((xsize, ysize))
    return img


if __name__ == '__main__':
    img_dir = r'.'
    reduce_dir = join(img_dir, 'reduce')
    if not os.path.exists(reduce_dir):
        os.mkdir(reduce_dir)
    for fname in listdir(img_dir):
        suffix = PurePosixPath(fname).suffix
        if suffix in ['.jpg', '.JPG', '.jpeg', '.JPEG', ]:
            with open(join(img_dir, fname), mode='rb') as fobj:
                stream = BytesIO(fobj.read())
                img = Image.open(stream).convert("RGB")
                img = reduce_pil_image(img)
                img.save(join(reduce_dir, fname.replace(suffix, '_reduce.jpg')))
