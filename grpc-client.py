import grpc
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc, InferResult
import cv2
from utils import *
from yolov5_utils import *
import time
import string


url = "192.168.1.198:8001"
model_name = "detect-plate"
labels = {}
input_name = "images"
c, w, h = 3, 640, 640
input_shape = (640, 640)
output_name = "output0"
source = "frame.jpg"
model_version = "1"
cur_id = 0

for index, char in enumerate(string.digits + string.ascii_lowercase):
    labels[index] = char
# init grpc
MAX_BATCH_SIZE = 8
channel_opt = [('grpc.max_send_message_length', MAX_BATCH_SIZE*4 * c * w * h),
               ('grpc.max_receive_message_length', MAX_BATCH_SIZE*4 * c * w * h)]
channel = grpc.insecure_channel(url, options=channel_opt)
grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)


# Load the image
ori_frame = cv2.imread(source)
frame = ori_frame.copy()

# Preprocess the image
preprocessed_image, rs_image = input_preprocess(frame, input_shape)
# stack image
images = [preprocessed_image] * 1
images = np.stack(images, axis=0)

# Prepare the input
t = time.time()
request = service_pb2.ModelInferRequest()
request.model_name = model_name
request.model_version = model_version
request.id = str(cur_id)

input = service_pb2.ModelInferRequest().InferInputTensor()
input.name = input_name
input.datatype = "FP32"
input.shape.extend([1, c, w, h])
request.inputs.extend([input])
print("Prepare input time: ", time.time() - t)
output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
output.name = output_name
request.outputs.extend([output])
request.raw_input_contents.extend([preprocessed_image.tobytes()])
response = grpc_stub.ModelInfer(request)
output = InferResult(response).as_numpy(output_name)
t = time.time()
det = non_max_suppression(output, conf_thres=0.5, iou_thres=0.5)
bboxes_dict = nms_to_bboxes(
    det, ori_frame=ori_frame, input_shape=input_shape)
print("NMS time: ", time.time() - t)
bboxes = bboxes_dict[0]

for box in bboxes:
    x1, y1, x2, y2, conf, cls = box
    cv2.rectangle(ori_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite("output.jpg", ori_frame)
