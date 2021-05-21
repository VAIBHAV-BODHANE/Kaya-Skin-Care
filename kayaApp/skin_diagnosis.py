"""
Mask R-CNN
Train on the skin robot dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    #Train a new model starting from pre-trained COCO weights
    python skin.py train --dataset=/home/.../mask_rcnn/data/skin/ --weights=coco

    #Train a new model starting from pre-trained ImageNet weights
    python skin.py train --dataset=/home/.../mask_rcnn/data/skin/ --weights=imagenet

    # Continue training the last model you trained. This will find
    # the last trained weights in the model directory.
    python skin.py train --dataset=/home/.../mask_rcnn/data/skin/ --weights=last

    #Detect and color splash on a image with the last model you trained.
    #This will find the last trained weights in the model directory.
    python skin.py splash --weights=last --image=/home/...../*.jpg

    #Detect and color splash on a video with a specific pre-trained weights of yours.
    python sugery.py splash --weights=/home/.../logs/mask_rcnn_skin_0030.h5  --video=/home/simon/Videos/Center.wmv
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from matplotlib import pyplot as plt
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from kayaApp.mrcnn.config import Config
from kayaApp.mrcnn import model as modellib, utils
from kayaApp.mrcnn import visualize
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class SkinConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "skin"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # Background + objects

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6


############################################################
#  Dataset
############################################################

class SkinDataset(utils.Dataset):
    def load_VIA(self, dataset_dir, subset, hc=False):
        """Load the skin dataset from VIA.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val or predict
        """
        # Add classes. We have only one class to add.
        self.add_class("skin", 1, "acne")
        self.add_class("skin", 2, "pigmented scar")
        self.add_class("skin", 3, "scar")
        self.add_class("skin", 4, "pih")
        self.add_class("skin", 5, "hyper pigmentation")
        self.add_class("skin", 6, "mole")
        self.add_class("skin", 7, "open pores")
        self.add_class("skin", 8, "melasma")
        if hc is True:
            for i in range(1,9):
                self.add_class("skin", i, "{}".format(i))
            self.add_class("skin", 1, "acne")
            self.add_class("skin", 2, "pigmented scar")
            self.add_class("skin", 3, "scar")
            self.add_class("skin", 4, "pih")
            self.add_class("skin", 5, "hyper pigmentation")
            self.add_class("skin", 6, "mole")
            self.add_class("skin", 7, "open pores")
            self.add_class("skin", 8, "melasma")


        # Train or validation dataset?
        assert subset in ["train", "val", "predict"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "train.json")))

        annotations = list(annotations.values())  # don't need the dict keys
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the circles that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            if type(a['regions']) is dict:
                circles = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                circles = [r['shape_attributes'] for r in a['regions']]
            # circles = [r['shape_attributes'] for r in a['regions']]
            names = [r['region_attributes'] for r in a['regions']]
            # load_mask() needs the image size to convert circles to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "skin",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                circles=circles,
                names=names)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a skin dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "skin":
            return super(self.__class__, self).load_mask(image_id)

        # Convert circles to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["circles"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["circles"]):
            # Get indexes of pixels inside the polygon and set them to 1
            if p['name']=='circle':
                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc= skimage.draw.circle(p['cy'],p['cx'], p['r'])
                mask[rr, cc, i] = 1
            elif p['name']=='ellipse':
                rr, cc= skimage.draw.ellipse(p['cy'],p['cx'],p['ry'],p['rx'])
                mask[rr, cc, i] = 1

        # Assign class_ids by reading class_names
        class_ids = np.zeros([len(info["circles"])])
        # In the skin dataset, pictures are labeled with name 'a' and 'r' representing arm and ring.
        for i, p in enumerate(class_names):
        #"name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}
            if p['skin'] == 'acne':
                class_ids[i] = 1
            elif p['skin'] == 'pigmented scar':
                class_ids[i] = 2
            elif p['skin'] == 'scar':
                class_ids[i] = 3
            elif p['skin'] == 'pih':
                class_ids[i] = 4
            elif p['skin'] == 'hyper pigmentation':
                class_ids[i] = 5
            elif p['skin'] == 'mole':
                class_ids[i] = 6
            elif p['skin'] == 'open pores':
                class_ids[i] = 7
            elif p['skin'] == 'melasma':
                class_ids[i] = 8

            #assert code here to extend to other labels
        class_ids = class_ids.astype(int)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "skin":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_mask_hc(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a skin dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "skin":
            return super(self.__class__, self).load_mask(image_id)

        # Convert circles to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        #"name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["circles"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["circles"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # Assign class_ids by reading class_names
        class_ids = np.zeros([len(info["circles"])])
        # In the skin dataset, pictures are labeled with name 'a' and 'r' representing arm and ring.
        for i, p in enumerate(class_names):
            if p['name'] == 'arm':
                class_ids[i] = 14
            elif p['name'] == 'error':
                pass
            else:
                class_ids[i] = int(p['name'])
            #assert code here to extend to other labels
        class_ids = class_ids.astype(int)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids

def train(model, *dic):
    """Train the model."""
    # Training dataset.
    dataset_train = SkinDataset()
    dataset_train.load_VIA(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SkinDataset()
    dataset_val.load_VIA(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedu le is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=60,
                layers='heads')


def GuidedFilt(img, r,eps):
    # eps = 0.002;

    I = np.double(img)
    I = I/255
    I2 = cv2.pow(I,2)
    mean_I = cv2.boxFilter(I,-1,((2*r)+1,(2*r)+1))
    mean_I2 = cv2.boxFilter(I2,-1,((2*r)+1,(2*r)+1))

    cov_I = mean_I2 - cv2.pow(mean_I,2)

    var_I = cov_I

    a = cv2.divide(cov_I,var_I+eps)
    b = mean_I - (a*mean_I)

    mean_a = cv2.boxFilter(a,-1,((2*r)+1,(2*r)+1))
    mean_b = cv2.boxFilter(b,-1,((2*r)+1,(2*r)+1))

    q = (mean_a * I) + mean_b

    return(np.uint8(q*255))

def before_after(image, mask,region):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    # gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    image = np.array(image)
    # boxes = region['rois']
    # N = boxes.shape[0]

    # for i in range(N):
    #     if not np.any(boxes[i]):
    #         # Skip this instance. Has no bbox. Likely lost in image cropping.
    #         continue
    #     y1, x1, y2, x2 = boxes[i]
        
    #     # radius = max(abs(x2-x1),abs(y2-y1))
    #     radius = 30
    #     x = abs(x2 - x1)
    #     y = abs(y2 - y1)
    #     print(x,y,radius,"ZZZZZZZZZZZZZZZZZZZ")
    #     image = blemishRemoval(image,x,y,radius)


    # image = GuidedFilt(image,4,0.002)
    # gd = GuidedFilt(image,6,0.004)
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        gd = GuidedFilt(image,4,0.008)
        splash = np.where(mask, gd, image).astype(np.uint8)
    else:
        splash = image
    return splash

def blemishRemoval(source,x, y,r):
    # Referencing global variables
    # global r, source
    # Action to be taken when left mouse button is pressed
    # Mark the center
    blemishLocation = (x,y)
    # print(blemishLocation,"blemishLocation")
    newX, newY = selectedBlemish(source,x,y,r)
    print(newX,newY,"NEWWWWWWWWWW")
    newPatch = source[newY:(newY+2*r), newX: (newX+2*r)]
    # newPatch = np.where(newPatch<0, 0, newPatch) 
    # cv2.imwrite("newpatch.png", newPatch) # Create mask for the new Patch
    mask = 255 * np.ones(newPatch.shape, newPatch.dtype)
    
    source = cv2.seamlessClone(newPatch, source, mask, blemishLocation, cv2.NORMAL_CLONE)
    # cv2.imshow("Blemish Removal App", source)
    # Action to be taken when left mouse button is released
    return source

def selectedBlemish(source,x,y,r):
    global i
    crop_img = source[y:(y+2*r), x:(x+2*r)]
    # i = 1 + 1
    # cv2.imwrite("blemish-"+ str(1) +".png", crop_img)
    return identifybestPatch(source,x,y,r)


def identifybestPatch(source,x,y,r):
    # Nearby Patches in all 8 directions
    patches={}
    key1tup = appendDictionary(source,x+2*r,y,r)
    patches['Key1'] = (x+2*r,y,key1tup[0],key1tup[1])

    key2tup = appendDictionary(source,x+2*r,y+r,r)
    patches['Key 2'] = (x+2*r,y+r, key2tup[0], key2tup[1])

    if x-2*r < 0:
        key3tup = appendDictionary(source,0,y,r)
        patches['Key 3'] = (0,y, key3tup[0], key3tup[1])
    else:
        key3tup = appendDictionary(source,x-2*r,y,r)
        patches['Key 3'] = (x-2*r,y, key3tup[0], key3tup[1])

    if x-2*r < 0 and y-r<0:
        key4tup = appendDictionary(source,0,0,r)
        patches['Key4'] = (0,0, key4tup[0], key4tup[1])
    elif x-2*r <0 and y-r >0:
        key4tup = appendDictionary(source,0,y-r,r)
        patches['Key4'] = (0,y-r, key4tup[0], key4tup[1])
    elif x-2*r>0 and y-r<0:
        key4tup = appendDictionary(source,x-2*r,0,r)
        patches['Key4'] = (x-2*r,0, key4tup[0], key4tup[1])
    elif x-2*r>0 and y-r>0:
        key4tup = appendDictionary(source,x-2*r,y-r,r)
        patches['Key4'] = (x-2*r,y-r, key4tup[0], key4tup[1])      

    key5tup = appendDictionary(source,x,y+2*r,r)
    patches['Key5'] = (x, y+2*r, key5tup[0], key5tup[1])

    key6tup = appendDictionary(source,x+r,y+2*r,r)
    patches['Key6'] = (x+r,y+2*r, key6tup[0], key6tup[1])

    if y-2*r<0:
        key7tup = appendDictionary(source,x,0,r)
        patches['Key7'] = (x, 0, key7tup[0], key7tup[1])
    else:
        key7tup = appendDictionary(source,x,y-2*r,r)
        patches['Key7'] = (x, y-2*r, key7tup[0], key7tup[1])

    if x-r<0 and y-2*r<0:
        key8tup = appendDictionary(source,0,0,r)
        patches['Key8'] = (0,0, key8tup[0], key8tup[1])
    elif x-r<0 and y-2*r>0:
        key8tup = appendDictionary(source,0,y-2*r,r)
        patches['Key8'] = (0, y-2*r, key8tup[0], key8tup[1])
    elif x-r>0 and y-2*r<0:
        key8tup = appendDictionary(source,x-r,0,r)
        patches['Key8'] = (x-r, 0, key8tup[0], key8tup[1])
    elif x-r>0 and y-2*r>0:
        key8tup = appendDictionary(source,x-r,y-2*r,r)
        patches['Key8'] = (x-r, y-2*r, key8tup[0], key8tup[1])

    findlowx = {}
    findlowy = {}
    for key,(x, y, gx, gy) in patches.items():
        findlowx[key] = gx

    for key,(x, y, gx, gy) in patches.items():
        findlowy[key] = gy

    y_key_min = min(findlowy.keys(), key=(lambda k: findlowy[k]))
    x_key_min = min(findlowx.keys(), key=(lambda k: findlowx[k]))

    print(patches,"PATCHHHHHHH")

    if x_key_min == y_key_min:
        return patches[x_key_min][0],patches[x_key_min][1]
    else:
        # print("Return x & y conflict, Can take help from FFT")
        return patches[x_key_min][0],patches[x_key_min][1]

def appendDictionary(source,x,y,r):
    print(x,y,r,"XXXXXXXXXXXXXXX")
    crop_img = source[y:(y+2*r), x: (x+2*r)]
    gradient_x, gradient_y = sobelfilter(crop_img)
    return gradient_x, gradient_y

def sobelfilter(crop_img):
    print(type(crop_img),"RRRRRRRR")
    sobelx64f = cv2.Sobel(crop_img, cv2.CV_64F, 1,0, ksize=3)
    abs_xsobel64f = np.absolute(sobelx64f)
    sobel_x8u = np.uint8(abs_xsobel64f)
    gradient_x = np.mean(sobel_x8u)

    sobely64f = cv2.Sobel(crop_img,cv2.CV_64F,0,1, ksize=3)
    abs_ysobel64f = np.absolute(sobely64f)
    sobel_y8u =  np.uint8(abs_ysobel64f)
    gradient_y = np.mean(sobel_y8u)

    return gradient_x, gradient_y


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_colors(image, number_of_colors, show_chart):

    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [colorsys.rgb_to_hsv(center_colors[i]) for i in counts.keys()]
#     print(center_colors[0],"ZZZZZZZZ")
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
#     np.array(colorsys.rgb_to_hsv(z[0][0],z[0][1],z[0][2]))
    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

    return rgb_colors


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None, out_dir=''):
    assert image_path or video_path

    class_names = ['BG', 'arm', 'ring']

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        # splash = color_splash(image, r['masks'])
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], making_image=True)
        file_name = 'splash.png'
        # Save output
        # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        # save_file_name = os.path.join(out_dir, file_name)
        # skimage.io.imsave(save_file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        # width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = 1600
        height = 1600
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.wmv".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        #For video, we wish classes keep the same mask in frames, generate colors for masks
        colors = visualize.random_colors(len(class_names))
        while success:
            print("frame: ", count)
            # Read next image
            plt.clf()
            plt.close()
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                # splash = color_splash(image, r['masks'])

                splash = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                     class_names, r['scores'], colors=colors, making_video=True)
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    os.makedirs('RESULTS')
    submit_dir = os.path.join(os.getcwd(), "RESULTS/")
    # Read dataset
    dataset = SkinDataset()
    dataset.load_VIA(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        canvas = visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'], detect=True)
            # show_bbox=False, show_mask=False,
            # title="Predictions",
            # detect=True)
        canvas.print_figure("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"][:-4]))
    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect rings and robot arms.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/home/simon/mask_rcnn/data/Skin",
                        help='Directory of the Skin dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/home/simon/logs/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SkinConfig()
    else:
        class InferenceConfig(SkinConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))


# dataset_dir = '/home/simon/deeplearning/mask_rcnn/data'
# dataset_train = SkinDataset()
# dataset_train.VIA(dataset_dir, "train")
# # dataset_train.prepare()
# a, b = dataset_train.load_mask(130)
# print(a.shape, b.shape)
# print(b)
