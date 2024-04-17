import grpc
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc, InferResult
import cv2
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
model_version = "1"
cur_id = 0

for index, char in enumerate(string.digits + string.ascii_lowercase):
    labels[index] = char


def main(index):
    # init grpc
    MAX_BATCH_SIZE = 8
    channel_opt = [('grpc.max_send_message_length', MAX_BATCH_SIZE*4 * c * w * h),
                   ('grpc.max_receive_message_length', MAX_BATCH_SIZE*4 * c * w * h)]
    channel = grpc.insecure_channel(url, options=channel_opt)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # init gRPC request
    request = service_pb2.ModelInferRequest()
    request.model_name = model_name
    request.model_version = model_version
    request.id = str(cur_id)

    # Prepare the input
    input = service_pb2.ModelInferRequest().InferInputTensor()
    input.name = input_name
    input.datatype = "FP32"
    input.shape.extend([1, c, w, h])
    request.inputs.extend([input])

    # prepare the output
    output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output.name = output_name
    request.outputs.extend([output])

    # Load the image
    path = "rtsp://192.168.1.198:8554/camera/head"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    count = 0
    old_time = time.time()
    num_frame = 0
    pre_time = []
    infer_time = []
    post_time = []
    while True:
        if time.time() - old_time > 1:
            print("FPS: ", count)
            print("Preprocess time: ", np.mean(pre_time))
            print("Infer time: ", np.mean(infer_time))
            print("Postprocess time: ", np.mean(post_time))
            pre_time = []
            infer_time = []
            post_time = []
            count = 0
            old_time = time.time()
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(path)
            continue
        num_frame += 1
        count += 1
        # Preprocess the image
        start_pre = time.time()
        preprocessed_image, rs_image = input_preprocess(frame, input_shape)
        images = np.expand_dims(preprocessed_image, axis=0)
        pre_time.append(time.time() - start_pre)
        # infer
        start_infer = time.time()
        request.raw_input_contents[:] = [images.tobytes()]
        response = grpc_stub.ModelInfer(request)
        infer_time.append(time.time() - start_infer)
        start_post = time.time()
        results = InferResult(
            response).as_numpy(output_name)
        nms_det = non_max_suppression(results, conf_thres=0.5, iou_thres=0.5)
        bboxes = nms_to_bboxes(nms_det, ori_frame=frame,
                               input_shape=input_shape)
        post_time.append(time.time() - start_post)
        first_bbox = bboxes[0]
        for bbox in first_bbox:
            x1, y1, x2, y2 = bbox[:4]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


main(0)
