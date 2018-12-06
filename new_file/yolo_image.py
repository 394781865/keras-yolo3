import sys,os
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import cv2

#制作适用于mtcnn的上半身检测数据集


def detect_img(yolo):
    image_path = '/home/lichen/keras-yolo3/images/'
    txt_data = open("/home/lichen/keras-yolo3/person.txt","w")
    for img in os.listdir(image_path):
        img_path = image_path + img
        image = Image.open(img_path)
        print(img_path)

        try:
           top, left, bottom, right = yolo.detect_image(image,img_path)
        except:
           continue
        finally:
           txt_data.write(str(img_path)+" ")
           txt_data.write(str(left)+" ")
           txt_data.write(str(top)+" ")
           txt_data.write(str(abs(right-left))+" ")
           txt_data.write(str(abs(bottom-top)))
           txt_data.write("\n")
    txt_data.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--model', type=str,default="/home/lichen/keras-yolo3/model_data/yolo.h5",
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )
    parser.add_argument(
        '--anchors', type=str,default="/home/lichen/keras-yolo3/model_data/yolo_anchors.txt",
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )
    parser.add_argument(
        '--classes', type=str,default="person",
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )
    parser.add_argument(
        '--gpu_num', type=int,default=0,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='/home/lichen/keras-yolo3/images/',
        help = "Video input path"
    )
    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    FLAGS = parser.parse_args()
    detect_img(YOLO(**vars(FLAGS)))




