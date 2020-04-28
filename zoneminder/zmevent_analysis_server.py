#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import cgi
import logging
from systemd.journal import JournalHandler
#import json
import sys
import os

# This is running from a git clone, not really installed
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from zmevent_config import DateSafeJsonEncoder, LOG_PATH_AS
from zmevent_image_analysis import YoloAnalyzer
from zmevent_models import MonitorZone

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

FORMAT = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, filename=LOG_PATH_AS)
logger = logging.getLogger()
logger.addHandler(JournalHandler())

ANALYZERS = [YoloAnalyzer]

YOLO_CFG_PATH = os.environ.get('YOLO_CFG_PATH','/zoneminder/cache/yolo')

OPENVINO_DEVICE = os.environ.get('OPENVINO_DEVICE', 'MYRIAD')

class OpenvinoYoloModel:

    def __init__(self):
        print('Before calling socket.listen()')
        self._ensure_configs()

        logger.info('[OpenvinoYoloModel] Instantiating YOLO3 Detector...')

        plugin = IEPlugin(device=OPENVINO_DEVICE)
        net = IENetwork(model = self._config_path('frozen_darknet_yolov3_model.xml'), 
            weights = self._config_path('frozen_darknet_yolov3_model.bin'))
        self._input_blob = next(iter(net.inputs))
        #self._net = plugin.load(network=net, config={"VPU_LOG_LEVEL": "LOG_DEBUG"})
        self._net = plugin.load(network=net)

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

    def _get_params(self):
        return (self._input_blob, self._net)   

class ZMEventAnalysisServer(BaseHTTPRequestHandler):

    def __init__(self, net_params, *args):
        self.net_params = net_params
        BaseHTTPRequestHandler.__init__(self, *args)

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write('{"status": "ok"}'.encode('utf-8'))

    def do_POST(self):
        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
        # refuse to receive non-json content
        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            return
        # read the message and convert it into a python dictionary
        length = int(self.headers.get('content-length'))
        message = json.loads(self.rfile.read(length).decode())

        response = self.analyze_event(message)

        # send the message back
        self._set_headers()
        self.wfile.write(
            json.dumps(response, cls=DateSafeJsonEncoder).encode('utf-8')
        )

    def analyze_event(self, msg):
        """
        returns a list of ObjectDetectionResult instances

        Sample event:

        {
            "EventId": 192843,
            "monitor_zones": {
                "36": {
                    "Type": "Active",
                    "Name": "DrivewayFar",
                    "point_list": [
                        [781, 264],
                        [1128, 412],
                        [877, 491],
                        [648, 297]
                    ],
                    "MonitorId": 9,
                    "Id": 36
                },
            },
            "hostname": "guarddog",
            "frames": {
                "119": "/usr/share/zoneminder/www/events/9/19/06/30/13/34/19/00119-capture.jpg"
            }
        }
        """
        logger.info(
            'Received analysis request for %s Event %s - %d frames',
            msg['hostname'], msg['EventId'], len(msg['frames'])
        )
        results = []
        for a in ANALYZERS:
            logger.debug('Running object detection with: %s', a)
            cls = a(
                {
                    x: MonitorZone(**msg['monitor_zones'][x])
                    for x in msg['monitor_zones'].keys()
                },
                msg['hostname'],
                self.net_params
            )
            for frameid, framepath in msg['frames'].items():
                res = cls.analyze(
                    msg['EventId'],
                    frameid,
                    framepath
                )
                results.append(res)
        logger.info(
            'Analysis for %s Event %s complete; returning %d results',
            msg['hostname'], msg['EventId'], len(results)
        )
        return results


def run():
    
    net_model = OpenvinoYoloModel()

    def handler(*args):
        ZMEventAnalysisServer(net_model._get_params(), *args)

    server_address = ('0.0.0.0', 8008)
    #httpd = HTTPServer(server_address, ZMEventAnalysisServer)

    httpd = HTTPServer(server_address, handler)
    print('Starting ZMEventAnalysisServer on port 8008...')
    logger.info('Starting ZMEventAnalysisServer on port 8008...')
    httpd.serve_forever()


if __name__ == "__main__":
    run()
