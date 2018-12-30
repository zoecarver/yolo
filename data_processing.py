import cv2
import numpy as np

from glob import glob
import xml.etree.ElementTree as ET

from keras.utils import Sequence

anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
anchors = [(0, 0, anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]

labels = ['pottedplant', 'bottle', 'bird', 'cow', 'sofa', 'tvmonitor', 'person', 'chair', 'aeroplane', 'dog', 'diningtable', 'train', 'head', 'hand', 'horse', 'cat', 'bicycle', 'boat', 'car', 'sheep', 'bus', 'motorbike', 'foot']

def find(lst, key, value):
    for i, dic in enumerate(lst):
        if dic[key] == value:
            return i
    return None


def get_objects(file):    
    tree = ET.parse(file)
    objects = []
    filename = None

    for element in tree.iter():
        if 'filename' in element.tag:
            filename = element.text
        if 'object' in element.tag or 'part' in element.tag:
            obj = {}

            for attr in list(element):
                if 'name' in attr.tag:
                    obj['name'] = attr.text
                    objects += [obj]

                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            obj['xmin'] = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            obj['ymin'] = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            obj['xmax'] = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            obj['ymax'] = int(round(float(dim.text)))

    return objects, filename


def parse_annotation(dir, x_path='JPEGImages/', y_path='Annotations/', img_ext='.jpg', annotations_glob=None):
    if dir[-1] != '/':
        dir += '/'

    if annotations_glob is None:
        annotations_glob = glob(dir + y_path + '*')

    images = []
    annotations = []

    count = 0
    length = len(annotations_glob)
    print('length: %f' % length)
    for file in annotations_glob:
        if count % 1000 == 0:
            print('percent complete: %f' % (count / length))

        base_path = file.split('/')[-1].split('.')[0]
        image_path = dir + x_path + base_path + img_ext
        annotation_path = dir + y_path + base_path + '.xml'

        boxes, filename = get_objects(annotation_path)
        image = cv2.imread(dir + x_path + filename)

        images += [image]
        annotations += [boxes]
        
        count += 1
        
    return images, annotations


def train_split(x, y, split_size=0.8):
    train_size = int(len(x) * split_size)

    return (x[:train_size], y[:train_size]), (x[train_size:], y[train_size:])


def overlap(a, b):
    '''
    get overlap of two boxes
    :param: a - box 1 min, box 1 max
    :param: b - box 2 min, box 2 max
    :returns: overlap
    '''

    x1, x2 = a
    x3, x4 = b

    if x3 < x1:
        if x4 < x1:
            return 0
        return min(x2, x4) - x1

    if x2 < x3:
        return 0
    return min(x2, x4) - x3


# boxes are in format: (xmin, ymin, xmax, ymax)
def get_iou(box_a, box_b):
    w_intersect = overlap([box_a[0], box_a[2]], [box_b[0], box_b[2]]) # compare x values (width)
    h_intersect = overlap([box_a[1], box_a[3]], [box_b[1], box_b[3]])

    intersect = w_intersect * h_intersect

    w1, h1 = box_a[2] - box_a[0], box_a[3] - box_a[1]
    w2, h2 = box_b[2] - box_b[0], box_b[3] - box_b[1]

    union = w1 * h1 + w2 * h2 - intersect
    return intersect / union


def get_best_anchor_index(box):
    max_iou = -1
    best_anchor_index = -1

    # boxes & anchors are in format: (xmin, ymin, xmax, ymax)
    for i, anchor in enumerate(anchors):
        iou = get_iou(box, anchor)

        if iou > max_iou:
            best_anchor_index = i
            max_iou = iou

    return best_anchor_index


def format_boxes(boxes, original_image_shape, new_image_shape=(416, 416)):
    # original_image_shape = original_image_shape[np.newaxis]

    original_image_shape = np.array(original_image_shape)
    new_image_shape = np.array(new_image_shape)
    shape_diff = original_image_shape / new_image_shape

    # format as class, x_min, y_min, x_max, y_max
    boxes = [
        (0, box['xmin'], box['ymin'], box['xmax'], box['ymax']) for box in boxes # TODO: `0` is a hack for class (box['name'])
    ]
    boxes = np.array([boxes]) # expand

    # generate center, width, and height
    boxes_xy = [(box[:, 3:5] + box[:, 1:3]) / 2 for box in boxes]
    boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]

    # scale to image shape
    boxes_xy = [box_xy / (416 / 13) for box_xy in boxes_xy]
    boxes_wh = [box_wh / (416 / 13) for box_wh in boxes_wh]

    # scale to image shape
    boxes_xy = [box_xy / shape_diff for box_xy in boxes_xy]
    boxes_wh = [box_wh / shape_diff for box_wh in boxes_wh]

    # reshape into (x, y, h, w, class)
    boxes = [
        np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)
    ][0]

    y = np.zeros((13, 13, 5, 5 + 1))

    for count, box in enumerate(boxes):
        grid_x = int(np.floor(box[0]))
        grid_y = int(np.floor(box[1]))

        if grid_x > 12:
            print('grid_x too big: ', grid_x)
            grid_x = 12

        if grid_y > 12:
            print('grid_y too big: ', grid_y)
            grid_y = 12

        # format (xmin, ymin, xmax, ymax)
        shifted_box = (0, 0, *box[2:4]) # we only want a point so we can compare it to the anchors
        best_anchor_index = get_best_anchor_index(shifted_box)

        y[grid_y, grid_x, best_anchor_index, 0:4] = box[:4]
        y[grid_y, grid_x, best_anchor_index, 4] = 1. # objectness
        y[grid_y, grid_x, best_anchor_index, 5 + 0] = 1. # class (TODO: other classes `0` should be class index)

    return y

    # get max number of boxes so we can pad accordingly
    max_boxes = 0
    for box in boxes:
        if box.shape[0] > max_boxes:
            max_boxes = box.shape[0]

    # pad the data
    for i, box in enumerate(boxes):
        box_shape = box.shape[0]

        if box_shape < max_boxes:
            padding = np.zeros((max_boxes - box_shape, 5), dtype=np.float32)
            boxes[i] = np.vstack(box, padding)

    return np.array(boxes)


def _format_boxes(boxes, original_shape=(416, 416)):
    y_batch = np.zeros((13, 13, 5, 5 + 80))

    img_h, img_w = original_shape

    for obj in boxes:
        if obj['xmax'] < obj['xmin'] or obj['ymax'] < obj['ymin']:
            print('invalid values:', obj)
            continue

        for attr in ['xmin', 'xmax']:
            obj[attr] = int(obj[attr] / (img_w / 416.))
            # obj[attr] = max(min(obj[attr], 416.), 0.)

        for attr in ['ymin', 'ymax']:
            obj[attr] = int(obj[attr] / (img_h / 416.))
            # obj[attr] = max(min(obj[attr], 416.), 0.)

        center_x = .5 * (obj['xmin'] + obj['xmax'])
        center_x = center_x / (416. / 13.)
        center_y = .5 * (obj['ymin'] + obj['ymax'])
        center_y = center_y / (416. / 13.)

        grid_x = int(np.floor(center_x))
        grid_y = int(np.floor(center_y))

        if grid_x < 13. and grid_y < 13.:
            obj_indx = labels.index(obj['name'])

            center_w = (obj['xmax'] - obj['xmin']) / (
                        416. / 13.)  # unit: grid cell
            center_h = (obj['ymax'] - obj['ymin']) / (
                        416. / 13.)  # unit: grid cell

            # find the anchor that best predicts this box
            best_anchor = -1
            max_iou = -1

            shifted_box = (0, 0, center_w, center_h)

            for i in range(len(anchors)):
                anchor = anchors[i]
                iou = get_iou(shifted_box, anchor)

                if max_iou < iou:
                    best_anchor = i
                    max_iou = iou

            box = [
                center_x,
                center_y,
                center_w,
                center_h]

            # assign ground truth x, y, w, h, confidence and class probs to y_batch
            y_batch[grid_y, grid_x, best_anchor, 0:4] = box
            y_batch[grid_y, grid_x, best_anchor, 4] = 1.
            y_batch[grid_y, grid_x, best_anchor, 5 + obj_indx] = 1.
        else:
            print('bad coordinates')
    
    return y_batch


def get_data(data_dir, images=None, annotations=None):
    images, annotations = parse_annotation(data_dir)
    
    labels = []
    index = 0
    for image, boxes in zip(images, annotations):
        original_shape = image.shape[:2]
        boxes = _format_boxes(boxes, original_shape=original_shape)
        annotations[index] = boxes
        image = cv2.resize(image, (416, 416))
        images[index] = image

        index += 1

    print('final labels:')
    print(labels)
    # return images, annotations
    return train_split(images, annotations)


def foo(): return 12


class VOCDataGenerator(Sequence):

    def __init__(self, annotations_glob, base_data_dir='VOCdevkit/VOC2012', batch_size=32):
        self.annotations_glob = annotations_glob
        self.batch_size = batch_size
        self.base_data_dir = base_data_dir
        self.count = 0

    def __len__(self):
        '''number of batches per epoch'''
        return int(np.floor(len(self.annotations_glob) / self.batch_size))

    def __getitem__(self, index):
        ''''generate one batch of data'''

        # Generate indexes of the batch
        self.count += self.batch_size
        batch = self.annotations_glob[self.count - self.batch_size: self.count]

        images, annotations = parse_annotation(self.base_data_dir, annotations_glob=batch)

        index = 0
        for image, boxes in zip(images, annotations):
            original_shape = image.shape[:2]
            boxes = _format_boxes(boxes, original_shape=original_shape)
            annotations[index] = boxes
            image = cv2.resize(image, (416, 416))
            images[index] = image

            index += 1

        train_x = np.array(images, dtype='float32')  # hack
        train_x = train_x[..., ::-1]

        train_y = np.array(annotations, dtype='float32')
        dummy = np.zeros((len(train_x), 1, 1, 1, 10, 4)) # 10 = TRUE_BOX_BUFFER

        return [train_x, dummy], train_y

    def on_epoch_end(self):
        # re-shuffle the glob
        np.random.shuffle(self.annotations_glob)
        self.count = 0

# (train_x, train_y), (test_x, test_y) = get_data('data')
