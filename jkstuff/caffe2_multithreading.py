from caffe2.python import workspace, model_helper

import numpy as np
import threading

def main():

    data = np.random.rand(16, 100).astype(np.float32)
    label = (np.random.rand(16) * 10).astype(np.int32)

    workspace.FeedBlob("data", data)
    workspace.FeedBlob("label", label)

    m = model_helper.ModelHelper(name="my first net")

    weight = m.param_init_net.XavierFill([], "fc_w", shape=[10, 100])
    bias = m.param_init_net.ConstantFill([], "fc_b", shape=[10, ])

    fc_l = m.net.FC(["data", "fc_w", "fc_b"], "fcl")
    pred = m.net.Sigmoid(fc_l, "pred")
    softmax, loss = m.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])

    workspace.RunNetOnce(m.param_init_net)
    workspace.CreateNet(m.net)

    for _ in range(100):
        data = np.random.rand(16, 100).astype(np.float32)
        label = (np.random.rand(16) * 10).astype(np.int32)
        workspace.FeedBlob("data", data)
        workspace.FeedBlob("label", label)
        workspace.RunNet(m.name, 10)

    print("{} in {} fetched blob softmax:\n{}".format(workspace.CurrentWorkspace(), threading.current_thread().name, workspace.FetchBlob("softmax")))
    print("{} in {} fetched blob loss:\n{}".format(workspace.CurrentWorkspace(), threading.current_thread().name, workspace.FetchBlob("loss")))

if __name__ == '__main__':
    main()
