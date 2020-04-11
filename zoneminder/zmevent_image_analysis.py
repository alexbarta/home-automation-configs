import os
import time
import math
import logging
from systemd.journal import JournalHandler
import numpy as np
import configparser, ast
from textwrap import dedent
import requests
from shapely.geometry.polygon import LinearRing, Polygon
from zmevent_config import IGNORED_OBJECTS
from zmevent_models import DetectedObject, ObjectDetectionResult


try:
    import cv2
except ImportError:
    raise SystemExit(
        'could not import cv2 - please "pip install opencv-python"'
    )
try:
    from openvino.inference_engine import IENetwork, IEPlugin
except ImportError:
    raise SystemExit(
        'could not import openvino libraries :('
    )

configAS = configparser.ConfigParser()

logger = logging.getLogger(__name__)
logger.addHandler(JournalHandler())

#: Path on disk where darknet yolo configs/weights will be stored
YOLO_CFG_PATH = os.environ.get('YOLO_CFG_PATH','/zoneminder/cache/yolo')
#YOLO_ALT_CFG_PATH = '/var/cache/zoneminder/yolo-alt'
#OPENVINO_DEVICE = os.environ.get('OPENVINO_DEVICE', 'MYRIAD')

configAS.read(YOLO_CFG_PATH+"/config-analysis-server.ini")

#CAMERA_DEFAULT_WIDTH = os.environ.get('CAMERA_DEFAULT_WIDTH', 1280)
#CAMERA_DEFAULT_HEIGHT = os.environ.get('CAMERA_DEFAULT_HEIGHT', 720)
CAMERA_DEFAULT_WIDTH = os.environ.get('CAMERA_DEFAULT_WIDTH', 1920)
CAMERA_DEFAULT_HEIGHT = os.environ.get('CAMERA_DEFAULT_HEIGHT', 1080)

ZM_DATA_PATH_PREFIX = os.environ.get('ZM_DATA_PATH_PREFIX', '/zoneminder/cache')

MODEL_INPUT_SIZE = configAS.getint("yolo", "MODEL_INPUT_SIZE")

ANCHORS = ast.literal_eval(configAS.get("yolo", "ANCHORS"))

LABELS = tuple(ast.literal_eval(configAS.get("yolo", "LABELS")))
#LABELS = ["person"]

yolo_scale_13 = configAS.getint("yolo", "yolo_scale_13")
yolo_scale_26 = configAS.getint("yolo", "yolo_scale_26")
yolo_scale_52 = configAS.getint("yolo", "yolo_scale_52")

#classes = configAS.getint("yolo", "classes")
classes = len(LABELS)
coords = configAS.getint("yolo", "coords")
num = configAS.getint("yolo", "num")

def EntryIndex(side, lcoords, lclasses, location, entry):
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)

class DetectionObject():
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence

    def _results(self):
        #return (LABELS[self.class_id], self.confidence.astype(float), 
        #    (float(self.xmax), float(self.ymax), float(self.xmin), float(self.ymin)))
        return (LABELS[self.class_id], self.confidence.astype(float), 
            (self.xmin, self.ymin, self.xmax, self.ymax))


class suppress_stdout_stderr(object):
    """
    Context manager to do "deep suppression" of stdout and stderr.

    from: https://stackoverflow.com/q/11130156/211734

    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class ImageAnalyzer(object):
    """
    Base class for specific object detection algorithms/packages.
    """

    def __init__(self, monitor_zones, hostname, net_params):
        """
        Initialize an image analyzer.

        :param monitor_zones: dict of string zone names to MonitorZone objects,
          for the monitor this event happened on
        :type monitor_zones: dict
        """
        self._monitor_zones = monitor_zones
        self._hostname = hostname
        self._net_params = net_params

    def analyze(self, event_id, frame_id, frame_path):
        """
        Analyze a frame; return an ObjectDetectionResult.
        """
        raise NotImplementedError('Implement in subclass!')


class YoloAnalyzer(ImageAnalyzer):
    """Object detection using yolo34py and yolov3-tiny"""

    def __init__(self, monitor_zones, hostname, net_params):
        super(YoloAnalyzer, self).__init__(monitor_zones, hostname, net_params)
        #self._ensure_configs()
        logger.info('[YoloAnalyzer] Instantiating YOLO3 Detector...')
        logger.info("classes: %s      labels: %s", classes, LABELS)
        logger.info("coords: %s", coords)
        logger.info("num: %s", num)
        #with suppress_stdout_stderr():
            #self._net = Detector(
            #    bytes(self._config_path("yolov3.cfg"), encoding="utf-8"),
            #    bytes(self._config_path("yolov3.weights"), encoding="utf-8"),
            #    0,
            #    bytes(self._config_path("coco.data"), encoding="utf-8")
            #)
        #plugin = IEPlugin(device=OPENVINO_DEVICE)
        #net = IENetwork(model = self._config_path('frozen_darknet_yolov3_model.xml'), 
        #    weights = self._config_path('frozen_darknet_yolov3_model.bin'))
        #self._input_blob = next(iter(net.inputs))
        self._input_blob = self._net_params[0]
        #self._net = plugin.load(network=net, config={"VPU_LOG_LEVEL": "LOG_DEBUG"})
        self._net = self._net_params[1] #, config={"VPU_LOG_LEVEL": "LOG_DEBUG"})

        logger.info('Done instantiating YOLO3 Detector.')

    def _ensure_configs(self):
        """Ensure that yolov3-tiny configs and data are in place."""
        # This uses the yolov3-tiny, because I only have a 1GB GPU
        if not os.path.exists(YOLO_CFG_PATH):
            raise SystemExit('I could not find YOLO_CFG_PATH: %s' % YOLO_CFG_PATH)
        
        configs = ['frozen_darknet_yolov3_model.xml', 'frozen_darknet_yolov3_model.bin']
      
        for file in configs:
            path = self._config_path(file)
            if not os.path.exists(path):
                raise SystemExit("Could not find file: ", path)

    def _config_path(self, f):
        return os.path.join(YOLO_CFG_PATH, f)

    def _prepare_image(self, image):
        # image resize
        logger.info("Default width : %s Default height : %s", CAMERA_DEFAULT_WIDTH ,CAMERA_DEFAULT_HEIGHT)
        resized_image = cv2.resize(image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation = cv2.INTER_CUBIC)
        canvas = np.full((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3), 128)
        #canvas_model_minus_height = ( MODEL_INPUT_SIZE - MODEL_INPUT_SIZE )
        #canvas_model_minus_width = ( MODEL_INPUT_SIZE - MODEL_INPUT_SIZE )
        #canvas[canvas_model_minus_height // 2 : canvas_model_minus_height // 2 + CAMERA_DEFAULT_HEIGHT, 
        #    canvas_model_minus_width // 2 : canvas_model_minus_width // 2 + CAMERA_DEFAULT_WIDTH,  :] = resized_image
        canvas[0:MODEL_INPUT_SIZE, 0:MODEL_INPUT_SIZE, :] = resized_image
        canvas = canvas[np.newaxis, :, :, :]     # Batch size axis add  
        return canvas.transpose((0, 3, 1, 2))  # NHWC to NCHW
             

    def do_image_yolo(self, event_id, frame_id, fname, detected_fname):
        """
        Analyze a single image using yolo34py.

        :param event_id: the EventId being analyzed
        :type event_id: int
        :param frame_id: the FrameId being analyzed
        :type frame_id: int
        :param fname: path to input image
        :type fname: str
        :param detected_fname: file path to write object detection image to
        :type detected_fname: str
        :return: yolo3 detection results
        :rtype: list of DetectedObject instances
        """
        fname = fname.replace('/var/cache/zoneminder', ZM_DATA_PATH_PREFIX)
        logger.info('Analyzing: %s', fname)
        img = cv2.imread(fname)
        #img2 = Image(img)
        prepimg = self._prepare_image(img)
        results = self._net.infer(inputs={self._input_blob: prepimg})
        logger.debug('Raw Results: %s', results)
        #results = self._net.detect(img2, thresh=0.2, hier_thresh=0.3, nms=0.4)

        objects = []

        for output in results.values():
            objects = self.parseYOLOV3Output(output, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 
                CAMERA_DEFAULT_HEIGHT, CAMERA_DEFAULT_WIDTH, 0.4, objects)
        logger.info("Found objects: %s", objects)

        retval = {'detections': [], 'ignored_detections': []}
        label=''

        for cat, score, bounds in objects:
            if not isinstance(cat, str):
                cat = cat.decode()
            x, y, w, h = bounds
            logger.info('DEBUG: cat,x,y,w,h,score %s,%d,%d,%d,%d,%.2f',cat,x,y,w,h,score)
            zones = self._zones_for_object(x, y, w, h)
            logger.info('Checking IgnoredObject filters for detections...')
            matched_filters = [
                foo.name for foo in IGNORED_OBJECTS.get(self._hostname, [])
                if foo.should_ignore(cat, x, y, w, h, zones, score)
            ]
            if len(matched_filters) > 0:
                # object should be ignored
                logger.info(
                    'Event %s Frame %s: Ignoring %s (%.2f) at %d,%d based on '
                    'filters: %s',
                    event_id, frame_id, cat, score, x, y, matched_filters
                )
                rect_color = (104, 104, 104)
                text_color = (111, 247, 93)
                retval['ignored_detections'].append(DetectedObject(
                    cat, zones, score, x, y, w, h, ignore_reason=matched_filters
                ))
            else:
                # object should not be ignored; add to result
                rect_color = (255, 0, 0)
                text_color = (255, 255, 0)
                retval['detections'].append(DetectedObject(
                    cat, zones, score, x, y, w, h
                ))
                label+='_'+cat
            #logger.info("x %s    y %s   w %s   h %s   int((w-x)*2) %s   int((h-y)*2) %s",x,y,w,h,str(int(((w-x)*2))),str(int((h-y)*2)))
            logger.info("x %s    y %s   w %s   h %s   int(w-x) %s   int(h-y) %s",x,y,w,h,str(int(w-x)),str(int(h-y)))
            cv2.rectangle(
                #img, (int(x - w / 2), int(y - h / 2)),
                #(int(x + w / 2), int(y + h / 2)), rect_color, thickness=2
                #img, (x,obj.ymin),(obj.xmax,obj.ymax),box_color,box_thickness
                #img, (int(x), int(y), int(w), int(h)), rect_color, thickness=2
                #img, (int(x), int(y), int((w-x)*2), int((h-y)*2)), rect_color, thickness=2
                img, (int(x), int(y), int(w-x), int(h-y)), rect_color, thickness=2
            )
            cv2.putText(
                img, '%s (%.2f)' % (cat, score),
                (int(x), int(y)),
                cv2.FONT_HERSHEY_COMPLEX, 1, text_color
            )
        detected_fname = detected_fname.replace('/var/cache/zoneminder', ZM_DATA_PATH_PREFIX)
        detected_fname = detected_fname.replace('.yolo3.jpg', '.yolo3'+label+'.jpg')

        logger.info('Writing: %s', detected_fname)
        cv2.imwrite(detected_fname, img)
        logger.info('Done with: %s', fname)
        return retval

    def _xywh_to_ring(self, x, y, width, height):
        points = [
            (x - (width / 2.0), y - (height / 2.0)),
            (x - (width / 2.0), y + (height / 2.0)),
            (x + (width / 2.0), y + (height / 2.0)),
            (x + (width / 2.0), y - (height / 2.0)),
            (x - (width / 2.0), y - (height / 2.0))
        ]
        return Polygon(LinearRing(points))

    def _zones_for_object(self, x, y, w, h):
        res = {}
        obj_polygon = self._xywh_to_ring(x, y, w, h)
        for zone in self._monitor_zones.values():
            if obj_polygon.intersects(zone.polygon):
                amt = (
                    obj_polygon.intersection(zone.polygon).area /
                    obj_polygon.area
                ) * 100
                res[zone.Name] = amt
        return res

    def analyze(self, event_id, frame_id, frame_path):
        _start = time.time()
        # get all the results
        output_path = frame_path.replace('.jpg', '.yolo3.jpg')
        res = self.do_image_yolo(event_id, frame_id, frame_path, output_path)
        _end = time.time()

        return ObjectDetectionResult(
            self.__class__.__name__,
            event_id,
            frame_id,
            frame_path,
            output_path,
            res['detections'],
            res['ignored_detections'],
            _end - _start
        )

    def parseYOLOV3Output(self, blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):
    
        out_blob_h = blob.shape[2]
        out_blob_w = blob.shape[3]
    
        side = out_blob_h
#        logger.debug('YOLO scale = %d     YOLO reso = %d     ANCHORS: %s', side, MODEL_INPUT_SIZE, ANCHORS)
        anchor_offset = 0
    
        if len(ANCHORS) == 18:   ## YoloV3
            if side == yolo_scale_13:
                anchor_offset = 2 * 6
            elif side == yolo_scale_26:
                anchor_offset = 2 * 3
            elif side == yolo_scale_52:
                anchor_offset = 2 * 0
    
        elif len(ANCHORS) == 12: ## tiny-YoloV3
            if side == yolo_scale_13:
                anchor_offset = 2 * 3
            elif side == yolo_scale_26:
                anchor_offset = 2 * 0
    
        else:                    ## ???
            if side == yolo_scale_13:
                anchor_offset = 2 * 6
            elif side == yolo_scale_26:
                anchor_offset = 2 * 3
            elif side == yolo_scale_52:
                anchor_offset = 2 * 0
    
        side_square = side * side
        output_blob = blob.flatten()
    
        for i in range(side_square):
            row = int(i / side)
            col = int(i % side)
            for n in range(num):
                obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords)
                box_index = EntryIndex(side, coords, classes, n * side * side + i, 0)
                scale = output_blob[obj_index]
                if (scale < threshold):
                    continue
                x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
                y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
                height = math.exp(output_blob[box_index + 3 * side_square]) * ANCHORS[anchor_offset + 2 * n + 1]
                width = math.exp(output_blob[box_index + 2 * side_square]) * ANCHORS[anchor_offset + 2 * n]

                for j in range(classes):
                    class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j)
                    prob = scale * output_blob[class_index]
                    if prob < threshold:
                        continue
                    logger.debug('Side,coords,classes,n,side_square %d,%d,%d,%d,%d',side,coords,classes,n,side_square)
                    logger.debug("Resized image coords: x: %d, y: %d, w: %d, h: %d", x, y, width, height)
                    obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                    objects.append(obj._results())
        return objects


class AlternateYoloAnalyzer(YoloAnalyzer):
    """
    This is used when I run from a script in a separate venv to compare CPU and
    GPU results.
    """

    def _config_path(self, f):
        return os.path.join(YOLO_ALT_CFG_PATH, f)

    def _ensure_configs(self):
        """Ensure that yolov3-tiny configs and data are in place."""
        # This uses the yolov3-tiny, because I only have a 1GB GPU
        if not os.path.exists(YOLO_ALT_CFG_PATH):
            logger.warning('Creating directory: %s', YOLO_ALT_CFG_PATH)
            os.mkdir(YOLO_ALT_CFG_PATH)
        configs = {
            'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/'
                          'master/cfg/yolov3.cfg',
            'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/'
                          'master/data/coco.names',
            'yolov3.weights': 'https://pjreddie.com/media/files/'
                              'yolov3.weights'
        }
        for fname, url in configs.items():
            path = self._config_path(fname)
            if os.path.exists(path):
                continue
            logger.warning('%s does not exist; downloading', path)
            logger.info('Download %s to %s', url, path)
            r = requests.get(url)
            logger.info('Writing %d bytes to %s', len(r.content), path)
            with open(path, 'wb') as fh:
                fh.write(r.content)
            logger.debug('Wrote %s', path)
        # coco.data is special because we change it
        path = self._config_path('coco.data')
        if not os.path.exists(path):
            content = dedent("""
            classes= 80
            train  = /home/pjreddie/data/coco/trainvalno5k.txt
            valid = %s
            names = %s
            backup = /home/pjreddie/backup/
            eval=coco
            """)
            logger.warning('%s does not exist; writing', path)
            with open(path, 'w') as fh:
                fh.write(content % (
                    self._config_path('coco_val_5k.list'),
                    self._config_path('coco.names')
                ))
            logger.debug('Wrote %s', path)

    def analyze(self, event_id, frame_id, frame_path):
        _start = time.time()
        # get all the results
        output_path = frame_path.replace('.jpg', '.yolo3alt.jpg')
        res = self.do_image_yolo(event_id, frame_id, frame_path, output_path)
        _end = time.time()
        return ObjectDetectionResult(
            self.__class__.__name__,
            event_id,
            frame_id,
            frame_path,
            output_path,
            res['detections'],
            res['ignored_detections'],
            _end - _start
        )
