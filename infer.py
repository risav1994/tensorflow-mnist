""" Reference URL: https://github.com/openvinotoolkit/cvat/blob/df175a856179f1c31ac2b15c80d98986a3d35acf/serverless/common/openvino/model_loader.py """
from openvino.inference_engine import IECore
import numpy as np

xTest = np.load("Data/images.npy")
yTest = np.load("Data/labels.npy")

core = IECore()
coreNet = core.read_network(
    model="saved_model.xml",
    weights="saved_model.bin"
)
net = core.load_network(coreNet, "CPU", num_requests=2)
count = 0
for idx, img in enumerate(xTest):
    inpBlobName = next(iter(coreNet.input_info))
    inputs = {inpBlobName: img}
    results = net.infer(inputs)
    outBlobName = next(iter(coreNet.outputs))
    if results[outBlobName][0] == yTest[idx]:
        count += 1
    print((results[outBlobName][0], yTest[idx]))

print(f"Accuracy: {100 * count / len(yTest)}")
