from vtk.inferrers.tensorflow import TensorFlowInferrer
from vtk.postprocessors.draw import DrawingPostprocessor
import cv2
import sys

def bbox_round(results):
    for i in range(len(results["detections"])):
        for j in range(len(results["detections"][i]["bbox"])):
            results["detections"][i]["bbox"][j] = int(results["detections"][i]["bbox"][j])
    return results

cap = cv2.VideoCapture(sys.argv[1])
out = cv2.VideoWriter(sys.argv[2], cv2.VideoWriter_fourcc('m','p','4','v'), cap.get(cv2.CAP_PROP_FPS), (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
inferrer = TensorFlowInferrer(sys.argv[3])
postprocessor = DrawingPostprocessor

while(cap.isOpened()):
    print(f"starting frame {cap.get(cv2.CAP_PROP_POS_FRAMES)} of {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    ret, frame = cap.read()
    results = inferrer.run(frame)
    results = bbox_round(results)
    if len(results["detections"]) > 0:
        print(f"detections: {len(results['detections'])}")
    output = postprocessor.run(None, frame, results)
    out.write(output)

cap.release()
print("done!")
