#Adapte from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts (preparation/json2labelImg.py, helpers/annotation.py )

from __future__ import print_function, absolute_import, division
import os, sys
import os, sys
import numpy as np
import h5py
try:
    import PIL.Image     as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)

import json
import argparse

from abc import ABCMeta, abstractmethod

class CsObjectType():
    """Type of an object"""
    POLY = 1  # polygon
    BBOX2D = 2  # bounding box
    BBOX3D = 3  # 3d bounding box
    IGNORE2D = 4  # 2d ignore region


class CsObject:
    """Abstract base class for annotation objects"""
    __metaclass__ = ABCMeta

    def __init__(self, objType):
        self.objectType = objType
        # the label
        self.label = ""

        # If deleted or not
        self.deleted = 0
        # If verified or not
        self.verified = 0
        # The date string
        self.date = ""
        # The username
        self.user = ""
        # Draw the object
        # Not read from or written to JSON
        # Set to False if deleted object
        # Might be set to False by the application for other reasons
        self.draw = True

    @abstractmethod
    def __str__(self): pass

    @abstractmethod
    def fromJsonText(self, jsonText, objId=-1): pass

    @abstractmethod
    def toJsonText(self): pass

    # def updateDate(self):
    #     try:
    #         locale.setlocale(locale.LC_ALL, 'en_US.utf8')
    #     except locale.Error:
    #         locale.setlocale(locale.LC_ALL, 'en_US')
    #     except locale.Error:
    #         locale.setlocale(locale.LC_ALL, 'us_us.utf8')
    #     except locale.Error:
    #         locale.setlocale(locale.LC_ALL, 'us_us')
    #     except Exception:
    #         pass
    #     self.date = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    # Mark the object as deleted
    def delete(self):
        self.deleted = 1
        self.draw = False

class CsBbox2d(CsObject):
    """Class that contains the information of a single annotated object as bounding box"""

    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.BBOX2D)
        # the polygon as list of points
        self.bbox_amodal_xywh = []
        self.bbox_modal_xywh = []

        # the ID of the corresponding object
        self.instanceId = -1
        # the label of the corresponding object
        self.label = ""

    def __str__(self):
        bboxAmodalText = ""
        bboxAmodalText += '[(x1: {}, y1: {}), (w: {}, h: {})]'.format(
            self.bbox_amodal_xywh[0], self.bbox_amodal_xywh[1],  self.bbox_amodal_xywh[2],  self.bbox_amodal_xywh[3])

        bboxModalText = ""
        bboxModalText += '[(x1: {}, y1: {}), (w: {}, h: {})]'.format(
            self.bbox_modal_xywh[0], self.bbox_modal_xywh[1], self.bbox_modal_xywh[2], self.bbox_modal_xywh[3])

        text = "Object: {}\n - Amodal {}\n - Modal {}".format(
            self.label, bboxAmodalText, bboxModalText)
        return text

    def setAmodalBox(self, bbox_amodal):
        # sets the amodal box if required
        self.bbox_amodal_xywh = [
            bbox_amodal[0],
            bbox_amodal[1],
            bbox_amodal[2] - bbox_amodal[0],
            bbox_amodal[3] - bbox_amodal[1]
        ]

    # access 2d boxes in [xmin, ymin, xmax, ymax] format
    @property
    def bbox_amodal(self):
        """Returns the 2d box as [xmin, ymin, xmax, ymax]"""
        return [
            self.bbox_amodal_xywh[0],
            self.bbox_amodal_xywh[1],
            self.bbox_amodal_xywh[0] + self.bbox_amodal_xywh[2],
            self.bbox_amodal_xywh[1] + self.bbox_amodal_xywh[3]
        ]

    @property
    def bbox_modal(self):
        """Returns the 2d box as [xmin, ymin, xmax, ymax]"""
        return [
            self.bbox_modal_xywh[0],
            self.bbox_modal_xywh[1],
            self.bbox_modal_xywh[0] + self.bbox_modal_xywh[2],
            self.bbox_modal_xywh[1] + self.bbox_modal_xywh[3]
        ]

    def fromJsonText(self, jsonText, objId=-1):
        # try to load from cityperson format
        if 'bbox' in jsonText.keys() and 'bboxVis' in jsonText.keys():
            self.bbox_amodal_xywh = jsonText['bbox']
            self.bbox_modal_xywh = jsonText['bboxVis']
        # both modal and amodal boxes are provided
        elif "modal" in jsonText.keys() and "amodal" in jsonText.keys():
            self.bbox_amodal_xywh = jsonText['amodal']
            self.bbox_modal_xywh = jsonText['modal']
        # only amodal boxes are provided
        else:
            self.bbox_modal_xywh = jsonText['2d']['amodal']
            self.bbox_amodal_xywh = jsonText['2d']['amodal']

        # load label and instanceId if available
        if 'label' in jsonText.keys() and 'instanceId' in jsonText.keys():
            self.label = str(jsonText['label'])
            self.instanceId = jsonText['instanceId']

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['instanceId'] = self.instanceId
        objDict['modal'] = self.bbox_modal_xywh
        objDict['amodal'] = self.bbox_amodal_xywh

        return objDict



class Annotation:
    """The annotation of a whole image (doesn't support mixed annotations, i.e. combining CsPoly and CsBbox2d)"""

    # Constructor
    def __init__(self, objType=CsObjectType.BBOX2D):

        # the list of objects
        self.objects = []
        # the camera calibration
        assert objType in CsObjectType.__dict__.values()
        self.objectType = objType

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def fromJsonText(self, jsonText):
        jsonDict = json.loads(jsonText)
        self.imgWidth = int(jsonDict['imgWidth'])
        self.imgHeight = int(jsonDict['imgHeight'])
        self.objects = []
        # load objects
        if self.objectType != CsObjectType.IGNORE2D:
            for objId, objIn in enumerate(jsonDict['objects']):
                obj = CsBbox2d()
                obj.fromJsonText(objIn, objId)
                self.objects.append(obj)

    # Read a json formatted file and return the annotation
    def fromJsonFile(self, jsonFile):
        if not os.path.isfile(jsonFile):
            print('Given json file not found: {}'.format(jsonFile))
            return
        with open(jsonFile, 'r') as f:
            jsonText = f.read()
            self.fromJsonText(jsonText)

    def toJsonFile(self, jsonFile):
        with open(jsonFile, 'w') as f:
            f.write(self.toJson())



def getPersonCarCountImagesFromLabels(person_annotations, car_annotations, out_img_size, img_height, img_width):

    person_allowed_labels = ['person', 'pedestrian', 'rider', 'sitting person', 'person (other)']
    person_objects = []
    car_objects = []
    for obj in person_annotations.objects:
        if obj.label in person_allowed_labels:
            amodal_area = obj.bbox_amodal_xywh[2] * obj.bbox_amodal_xywh[3]

            #Reject if visible area is less
            if (amodal_area < 50):
                continue

            person_objects.append(obj)
        else:
            continue

    for obj in car_annotations.objects:
        if obj.label == 'car':
            amodal_area = obj.bbox_amodal_xywh[2] * obj.bbox_amodal_xywh[3]

            #Reject if visible area is less
            if (amodal_area < 50):
                continue

            car_objects.append(obj)
        else:
            continue

    return getImageCoordinatesAndCountFromObjects(person_objects, car_objects, out_img_size, img_height, img_width)

def getImageCoordinatesAndCountFromObjects(person_objects, car_objects, out_img_size, img_height, img_width):
    label_dict = {} # count_tuple: #Array of image cordinates

    image_dict = {}
    for person_object in person_objects:
        xywh = person_object.bbox_amodal_xywh
        center = np.array([xywh[0] + xywh[2]//2, xywh[1] + xywh[3]//2])
        image_bottomleft = center - [out_img_size//2]
        image_topright   = center + [out_img_size//2]

        if image_bottomleft[0] < 0 or image_bottomleft[1] < 0:
            print ("Out of image " + str(image_bottomleft))
            continue

        if image_topright[0] > img_width or image_topright[1] > img_height:
            continue
        image_xywh = np.concatenate((image_bottomleft, image_topright))
        count_tuple = getCountOfPersonAndCarsinImage(image_xywh, person_objects, car_objects)

        if count_tuple not in label_dict:
            label_dict[count_tuple] = [image_xywh]
        else:
            label_dict[count_tuple].append(image_xywh)


    for car_object in car_objects:
        xywh = car_object.bbox_amodal_xywh
        center = np.array([xywh[0] + xywh[2]//2, xywh[1] + xywh[3]//2])
        image_bottomleft = center - [out_img_size//2]
        image_topright   = center + [out_img_size//2]

        if image_bottomleft[0] < 0 or image_bottomleft[1] < 0:
            print ("Out of image " + str(image_bottomleft))
            continue

        if image_topright[0] > img_width or image_topright[1] > img_height:
            continue
        image_xywh = np.concatenate((image_bottomleft, image_topright))
        count_tuple = getCountOfPersonAndCarsinImage(image_xywh, person_objects, car_objects)
        if count_tuple[0] > 0 or count_tuple[1] > 0:
            if count_tuple not in label_dict:
                label_dict[count_tuple] = [image_xywh]
            else:
                label_dict[count_tuple].append(image_xywh)

    return label_dict

def getCountOfPersonAndCarsinImage(image_xywh, person_objects, car_objects):
    person_count = 0
    car_count    = 0
    for person_object in person_objects:
        person_xywh = person_object.bbox_amodal
        dx = min(person_xywh[2], image_xywh[2]) - max(person_xywh[0], image_xywh[0])
        dy = min(person_xywh[3], image_xywh[3]) - max(person_xywh[1], image_xywh[1])

        if (dx >= 0) and (dy >= 0):
            intersection_area = dx*dy
        else:
            intersection_area = 0
        person_area = (person_xywh[2] - person_xywh[0]) * (person_xywh[3] - person_xywh[1])
        person_in_image = (intersection_area/person_area) > 0.5
        if person_in_image:
            person_count += 1

    for car_object in car_objects:
        car_xywh = car_object.bbox_amodal_xywh
        dx = min(car_xywh[0] + car_xywh[2], image_xywh[2]) - max(car_xywh[0], image_xywh[0])
        dy = min(car_xywh[1] + car_xywh[3], image_xywh[3]) - max(car_xywh[1], image_xywh[1])

        if (dx >= 0) and (dy >= 0):
            intersection_area = dx*dy
        else:
            intersection_area = 0
        car_area = (car_xywh[2]) * (car_xywh[3])

        car_in_image = (intersection_area/car_area) > 0.5
        if car_in_image:
            car_count += 1

    return person_count, car_count


def json2label(inJsonPerson, inJsonCar, out_img_size, img_height, img_width):
    person_annotations = Annotation()
    person_annotations.fromJsonFile(inJsonPerson)
    car_annotations = Annotation()
    car_annotations.fromJsonFile(inJsonCar)
    return getPersonCarCountImagesFromLabels(person_annotations, car_annotations, out_img_size, img_height, img_width)


def save_images(img, label_dict, file_name, city_count_root):
    for count_tuple, sub_images in label_dict.items():
        dir_path = city_count_root + '/' + str(str(count_tuple[0]) + str(count_tuple[1]))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for i, sub_image in enumerate(sub_images):
            cropped_img = img.crop(sub_image)
            cropped_img.save(dir_path+ '/' + file_name + '_'+str(i) + '.png')


def create_city_count_data(city_person_label, city_car_label, cityscapes_img, city_count_root, out_img_size):
    if city_person_label == '' or city_car_label == '' or cityscapes_img == '' or city_count_root=='':
        print ("Invalid path")
        return {}

    img = Image.open(cityscapes_img)
    img_width, img_height  = img.size
    label_dict = json2label(city_person_label, city_car_label, out_img_size, img_height, img_width)
    file_name = ''.join(os.path.basename(city_person_label).split('_')[0:3])
    save_images(img, label_dict, file_name, city_count_root)
    json_file = city_count_root + '/' + file_name + ".json"
    label_dict = convert_dict_to_json_compatible(label_dict)
    #save_data_dict_in_json(label_dict, json_file)
    data_dict = {file_name: label_dict}
    return data_dict

def save_data_dict_in_json(datadict, jsonfile):
    with open(jsonfile, "w") as outfile:
        json.dump(datadict, outfile)

def convert_dict_to_json_compatible(label_dict):
    string_dict = {}
    for tuple, value in label_dict.items():
        string_dict[str(tuple[0])+'_'+str(tuple[1])] = [l.tolist() for l in value]
    return string_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input options
    parser.add_argument('--city_gtBbox3d_root', default='',
                        help="The root directory of the gtBbox3d labels")
    parser.add_argument('--city_gtBboxCityPersons_root',
                        default='',
                        help="The root directory of the gtBboxCityPersons labels")
    parser.add_argument('--city_image_dataset_root',
                        default='',
                        help="The root directory of the cityscapes images dataset")
    parser.add_argument('--city_count_root',
                        default="",
                        help="The root directory of the generated city count dataset")

    parser.add_argument('--out_img_size',
                        default=254,
                        help="The output image size")
    args = parser.parse_args()

    city_person_label = args.city_gtBboxCityPersons_root
    city_car_label    = args.city_gtBbox3d_root
    cityscapes_img    = args.city_image_dataset_root
    city_count_root   = args.city_count_root
    out_img_size      = args.out_img_size

    if city_person_label == '' or city_car_label == '' or cityscapes_img == '' or city_count_root=='':
        print ("Invalid path")

    city_car_label_files = []
    for subdir, dirs, files in os.walk(city_car_label):
        for file in files:
            city_car_label_files.append(os.path.join(subdir, file))

    city_person_label_files = []
    for subdir, dirs, files in os.walk(city_person_label):
        for file in files:
            city_person_label_files.append(os.path.join(subdir, file))


    img_files = []
    for subdir, dirs, files in os.walk(cityscapes_img):
        for file in files:
            img_files.append(os.path.join(subdir, file))

    for file in img_files:
        base_names = os.path.basename(file).split('_')[0:3]
        search_name = base_names[0] + '_' + base_names[1] + '_' + base_names[2]
        # get 3d and city person labels with similar base name
        city_car_labels = []
        for city_car_label_file_path in city_car_label_files:
            if search_name in city_car_label_file_path:
                city_car_labels.append(city_car_label_file_path)
                break

        city_person_labels = []
        for city_person_label_file_path in city_person_label_files:
            if search_name in city_person_label_file_path:
                city_person_labels.append(city_person_label_file_path)
                break

        city_person_label_file = '' if len(city_person_labels) == 0 else city_person_labels[0]
        city_car_label_file = '' if len(city_car_labels) == 0 else city_car_labels[0]

        if city_car_label_file != '' and city_person_label_file != '':
            data_dict = create_city_count_data(city_person_label_file,
                                               city_car_label_file,
                                               file,
                                               city_count_root, out_img_size)
