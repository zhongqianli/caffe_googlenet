import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

def inception(net, pre_layer, conv1x1_num,
              conv3x3_reduce_num, conv3x3_num,
              conv5x5_reduce_num, conv5x5_num,
              maxpool3x3_proj1x1_num,
              name):
    # 1x1
    net.conv1x1 = L.Convolution(pre_layer, kernel_size=1, num_output=conv1x1_num,
                                weight_filler=dict(type="xavier"),
                                bias_filler=dict(type="constant", value=0),
                                param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                name="{0}_conv1x1".format(name))
    net.conv1x1_relu = L.ReLU(net.conv1x1, in_place=True,
                              name="{0}_conv1x1_relu".format(name))

    # 3x3
    net.conv3x3_reduce = L.Convolution(pre_layer, kernel_size=1, num_output=conv3x3_reduce_num,
                                weight_filler=dict(type="xavier"),
                                bias_filler=dict(type="constant", value=0),
                                param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                       name="{0}_conv3x3_reduce".format(name))
    net.conv3x3_reduce_relu = L.ReLU(net.conv3x3_reduce, in_place=True,
                                     name="{0}_conv3x3_reduce_relu".format(name))

    net.conv3x3 = L.Convolution(net.conv3x3_reduce_relu, kernel_size=3, num_output=conv3x3_num, pad=1,
                                       weight_filler=dict(type="xavier"),
                                       bias_filler=dict(type="constant", value=0),
                                       param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                name="{0}_".format(name))
    net.conv3x3_relu = L.ReLU(net.conv3x3, in_place=True,
                              name="{0}_conv3x3".format(name))

    # 5x5
    net.conv5x5_reduce = L.Convolution(pre_layer, kernel_size=1, num_output=conv5x5_reduce_num,
                                       weight_filler=dict(type="xavier"),
                                       bias_filler=dict(type="constant", value=0),
                                       param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                       name="{0}_conv5x5_reduce".format(name))
    net.conv5x5_reduce_relu = L.ReLU(net.conv5x5_reduce, in_place=True,
                                     name="{0}_conv5x5_reduce_relu".format(name))

    net.conv5x5 = L.Convolution(net.conv5x5_reduce_relu, kernel_size=5, num_output=conv5x5_num, pad=2,
                                       weight_filler=dict(type="xavier"),
                                       bias_filler=dict(type="constant", value=0),
                                       param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                name="{0}_conv5x5".format(name))
    net.conv5x5_relu = L.ReLU(net.conv5x5, in_place=True,
                              name="{0}_conv5x5_relu".format(name))

    # pool
    net.maxpool3x3 = L.Pooling(pre_layer, kernel_size=3, stride=1, pad=1, pool=P.Pooling.MAX,
                               name="{0}_maxpool3x3".format(name))
    net.maxpool3x3_proj1x1 = L.Convolution(net.maxpool3x3, kernel_size=1, num_output=maxpool3x3_proj1x1_num,
                                       weight_filler=dict(type="xavier"),
                                       bias_filler=dict(type="constant", value=0),
                                       param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                           name="{0}_maxpool3x3_proj1x1".format(name))
    net.maxpool3x3_proj1x1_relu = L.ReLU(net.maxpool3x3_proj1x1, in_place=True,
                                         name="{0}_maxpool3x3_proj1x1_relu".format(name))

    # concat
    net.inception_output = L.Concat(net.conv1x1_relu, net.conv3x3_relu, net.conv5x5_relu, net.maxpool3x3_proj1x1_relu,
                                    name="{0}_output".format(name))
    return net.inception_output

def create_googlenet(input_shape, classes=1000, deploy=False):
    net_name = "googlenet"
    data_root_dir = "/home/tim/datasets/cifar10.224x224/"
    if deploy:
        net_filename = "{0}_deploy.prototxt".format(net_name)
    else:
        net_filename = "{0}_train_test.prototxt".format(net_name)

    # net name
    with open(net_filename, "w") as f:
        f.write('name: "{0}"\n'.format(net_name))

    if deploy:
        net = caffe.NetSpec()
        """
        The conventional blob dimensions for batches of image data are 
        number N x channel K x height H x width W. Blob memory is row-major in layout, 
        so the last / rightmost dimension changes fastest. 
        For example, in a 4D blob, the value at index (n, k, h, w) is 
        physically located at index ((n * K + k) * H + h) * W + w.
        """
        # batch_size, channel, height, width
        net.data = L.Input(input_param=dict(shape=[dict(dim=list(input_shape))]))
    else:
        net = caffe.NetSpec()
        batch_size = 32
        lmdb = data_root_dir + "train_lmdb"
        net.data, net.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                                     transform_param=dict(mirror=True,
                                                          crop_size=224,
                                                          #mean_file=data_root_dir + "mean.binaryproto"),
                                                          mean_value=[104, 117, 123]),
                                     ntop=2, include=dict(phase=caffe_pb2.Phase.Value("TRAIN")))

        with open(net_filename, "a") as f:
            f.write(str(net.to_proto()))

        del net
        net = caffe.NetSpec()
        batch_size = 50
        lmdb = data_root_dir + "test_lmdb"
        net.data, net.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                                     transform_param=dict(mirror=False,
                                                          crop_size=224,
                                                          # mean_file=data_root_dir + "mean.binaryproto"),
                                                          mean_value=[104, 117, 123]),
                                     ntop=2, include=dict(phase=caffe_pb2.Phase.Value("TEST")))

    # padding = 'same', equal to pad = 1
    net.conv1_7x7_2s = L.Convolution(net.data, kernel_size=7, num_output=64, pad=3, stride=2,
                              weight_filler=dict(type="xavier"),
                              bias_filler=dict(type="constant", value=0),
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.conv1_7x7_2s_relu = L.ReLU(net.conv1_7x7_2s, in_place=True)
    net.conv1_maxpool1_3x3_2s = L.Pooling(net.conv1_7x7_2s_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    net.conv1_norm1 = L.LRN(net.conv1_maxpool1_3x3_2s, local_size=5, alpha=0.0001, beta=0.75)

    net.conv2_1x1_1v = L.Convolution(net.conv1_norm1, kernel_size=1, num_output=64,
                              weight_filler=dict(type="xavier"),
                              bias_filler=dict(type="constant", value=0),
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.conv2_1x1_1v_relu = L.ReLU(net.conv2_1x1_1v, in_place=True)
    net.conv2_3x3_1s = L.Convolution(net.conv2_1x1_1v_relu, kernel_size=3, num_output=192, pad=1,
                              weight_filler=dict(type="xavier"),
                              bias_filler=dict(type="constant", value=0),
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.conv2_3x3_1s_relu = L.ReLU(net.conv2_3x3_1s, in_place=True)
    net.conv2_norm2 = L.LRN(net.conv2_3x3_1s_relu, local_size=5, alpha=0.0001, beta=0.75)
    net.conv2_pool_3x3_2s = L.Pooling(net.conv2_norm2, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    # inception(3a)
    inception3a_output = inception(net=net, pre_layer=net.conv2_pool_3x3_2s, conv1x1_num=64,
                                      conv3x3_reduce_num=96, conv3x3_num=128, conv5x5_reduce_num=16,
                                      conv5x5_num=32, maxpool3x3_proj1x1_num=32,
                                   name="inception3a")
    # inception(3b)
    inception3b_output = inception(net=net, pre_layer=inception3a_output, conv1x1_num=128,
                                   conv3x3_reduce_num=128, conv3x3_num=192, conv5x5_reduce_num=32,
                                   conv5x5_num=96, maxpool3x3_proj1x1_num=64,
                                   name="inception3b")

    # max pool
    net.inception3_maxpool = L.Pooling(inception3b_output, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    # inception(4a)
    inception4a_output = inception(net=net, pre_layer=net.inception3_maxpool, conv1x1_num=192,
                                   conv3x3_reduce_num=96, conv3x3_num=208, conv5x5_reduce_num=16,
                                   conv5x5_num=48, maxpool3x3_proj1x1_num=64,
                                   name="inception4a")

    # loss1
    if not deploy:
        # avg pool
        net.loss1_avgpool5x5_3v = L.Pooling(inception4a_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)

        # conv1x1_1s
        net.loss1_conv1x1_1s = L.Convolution(net.loss1_avgpool5x5_3v, kernel_size=1, num_output=128,
                                                  weight_filler=dict(type="xavier"),
                                                  bias_filler=dict(type="constant", value=0.2),
                                                  param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
        net.loss1_conv1x1_1s_relu = L.ReLU(net.loss1_conv1x1_1s, in_place=True)

        net.loss1_fc1 = L.InnerProduct(net.loss1_conv1x1_1s_relu, num_output=1024,
                                          weight_filler=dict(type="xavier"),
                                          bias_filler=dict(type="constant", value=0),
                                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
        net.loss1_fc1_relu1 = L.ReLU(net.loss1_fc1, in_place=True)

        net.loss1_dropout = L.Dropout(net.loss1_fc1_relu1, dropout_param=dict(dropout_ratio=0.7),
                                      in_place=True)
        net.loss1_pred_fc = L.InnerProduct(net.loss1_dropout, num_output=classes,
                                              weight_filler=dict(type="xavier"),
                                              bias_filler=dict(type="constant", value=0),
                                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
        net.loss1 = L.SoftmaxWithLoss(net.loss1_pred_fc, net.label, loss_weight=0.3)
        net.loss1_accuracy_top_1 = L.Accuracy(net.loss1_pred_fc, net.label,
                              include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        net.loss1_accuracy_top_5 = L.Accuracy(net.loss1_pred_fc, net.label,
                                          include=dict(phase=caffe_pb2.Phase.Value('TEST')),
                                              accuracy_param=dict(top_k=5))

    # inception(4b)
    inception4b_output = inception(net=net, pre_layer=inception4a_output, conv1x1_num=160,
                                   conv3x3_reduce_num=112, conv3x3_num=224, conv5x5_reduce_num=24,
                                   conv5x5_num=64, maxpool3x3_proj1x1_num=64,
                                   name="inception4b")

    # inception(4c)
    inception4c_output = inception(net=net, pre_layer=inception4b_output, conv1x1_num=128,
                                   conv3x3_reduce_num=128, conv3x3_num=256, conv5x5_reduce_num=24,
                                   conv5x5_num=64, maxpool3x3_proj1x1_num=64,
                                   name="inception4c")

    # inception(4d)
    inception4d_output = inception(net=net, pre_layer=inception4c_output, conv1x1_num=112,
                                   conv3x3_reduce_num=144, conv3x3_num=288, conv5x5_reduce_num=32,
                                   conv5x5_num=64, maxpool3x3_proj1x1_num=64,
                                   name="inception4d")

    # loss2
    if not deploy:
        # avg pool
        net.loss2_avgpool5x5_3v = L.Pooling(inception4d_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)

        # conv1x1_1s
        net.loss2_conv1x1_1s = L.Convolution(net.loss2_avgpool5x5_3v, kernel_size=1, num_output=128,
                                                  weight_filler=dict(type="xavier"),
                                                  bias_filler=dict(type="constant", value=0.2),
                                                  param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
        net.loss2_conv1x1_1s_relu = L.ReLU(net.loss2_conv1x1_1s, in_place=True)

        net.loss2_fc1 = L.InnerProduct(net.loss2_conv1x1_1s_relu, num_output=1024,
                                          weight_filler=dict(type="xavier"),
                                          bias_filler=dict(type="constant", value=0),
                                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
        net.loss2_fc1_relu1 = L.ReLU(net.loss2_fc1, in_place=True)

        net.loss2_dropout = L.Dropout(net.loss2_fc1_relu1, dropout_param=dict(dropout_ratio=0.7),
                                      in_place=True)
        net.loss2_pred_fc = L.InnerProduct(net.loss2_dropout, num_output=classes,
                                              weight_filler=dict(type="xavier"),
                                              bias_filler=dict(type="constant", value=0),
                                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
        net.loss2 = L.SoftmaxWithLoss(net.loss2_pred_fc, net.label, loss_weight=0.3)
        net.loss2_accuracy_top_1 = L.Accuracy(net.loss2_pred_fc, net.label,
                                          include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        net.loss2_accuracy_top_5 = L.Accuracy(net.loss2_pred_fc, net.label,
                                          include=dict(phase=caffe_pb2.Phase.Value('TEST')),
                                          accuracy_param=dict(top_k=5))

    # inception(4e)
    inception4e_output = inception(net=net, pre_layer=inception4d_output, conv1x1_num=256,
                                   conv3x3_reduce_num=160, conv3x3_num=320, conv5x5_reduce_num=32,
                                   conv5x5_num=128, maxpool3x3_proj1x1_num=128,
                                   name="inception4e")

    # max pool
    net.inception4_maxpool = L.Pooling(inception4e_output, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    # inception(5a)
    inception5a_output = inception(net=net, pre_layer=net.inception4_maxpool, conv1x1_num=256,
                                   conv3x3_reduce_num=160, conv3x3_num=320, conv5x5_reduce_num=32,
                                   conv5x5_num=128, maxpool3x3_proj1x1_num=128,
                                   name="inception5a")

    # inception(5b)
    inception5b_output = inception(net=net, pre_layer=inception5a_output, conv1x1_num=384,
                                   conv3x3_reduce_num=192, conv3x3_num=384, conv5x5_reduce_num=48,
                                   conv5x5_num=128, maxpool3x3_proj1x1_num=128,
                                   name="inception5b")

    # avg pool
    net.avgpool7x7_s1 = L.Pooling(inception5b_output, kernel_size=7, stride=1, pool=P.Pooling.AVE)

    # dropout
    net.avgpool7x7_s1_dropout = L.Dropout(net.avgpool7x7_s1, dropout_param=dict(dropout_ratio=0.4), in_place=True)

    # pred fc
    net.loss3_pred_fc = L.InnerProduct(net.avgpool7x7_s1_dropout, num_output=classes,
                                      weight_filler=dict(type="xavier"),
                                      bias_filler=dict(type="constant", value=0),
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    # loss3
    if deploy:
        net.prob = L.Softmax(net.loss3_pred_fc)
    else:
        net.loss3 = L.SoftmaxWithLoss(net.loss3_pred_fc, net.label)
        net.loss3_accuracy_top_1 = L.Accuracy(net.loss3_pred_fc, net.label,
                                              include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        net.loss3_accuracy_top_5 = L.Accuracy(net.loss3_pred_fc, net.label,
                                              include=dict(phase=caffe_pb2.Phase.Value('TEST')),
                                              accuracy_param=dict(top_k=5))

    with open(net_filename, "a") as f:
        f.write(str(net.to_proto()))

if __name__ == "__main__":

    input_shape = [1, 3, 224, 224]
    classes = 10

    create_googlenet(input_shape=input_shape, classes=classes, deploy=False)
    create_googlenet(input_shape=input_shape, classes=classes, deploy=True)

    net_name = "googlenet"

    solver = caffe.SGDSolver("{0}_solver.prototxt".format(net_name))

    for k, v in solver.net.blobs.items():
        print(k, v.data.shape)

# caffe models googlenet layers' shapes
"""
data (32, 3, 224, 224)
label (32,)
label_data_1_split_0 (32,)
label_data_1_split_1 (32,)
label_data_1_split_2 (32,)
conv1/7x7_s2 (32, 64, 112, 112)
pool1/3x3_s2 (32, 64, 56, 56)
pool1/norm1 (32, 64, 56, 56)
conv2/3x3_reduce (32, 64, 56, 56)
conv2/3x3 (32, 192, 56, 56)
conv2/norm2 (32, 192, 56, 56)
pool2/3x3_s2 (32, 192, 28, 28)
pool2/3x3_s2_pool2/3x3_s2_0_split_0 (32, 192, 28, 28)
pool2/3x3_s2_pool2/3x3_s2_0_split_1 (32, 192, 28, 28)
pool2/3x3_s2_pool2/3x3_s2_0_split_2 (32, 192, 28, 28)
pool2/3x3_s2_pool2/3x3_s2_0_split_3 (32, 192, 28, 28)
inception_3a/1x1 (32, 64, 28, 28)
inception_3a/3x3_reduce (32, 96, 28, 28)
inception_3a/3x3 (32, 128, 28, 28)
inception_3a/5x5_reduce (32, 16, 28, 28)
inception_3a/5x5 (32, 32, 28, 28)
inception_3a/pool (32, 192, 28, 28)
inception_3a/pool_proj (32, 32, 28, 28)
inception_3a/output (32, 256, 28, 28)
inception_3a/output_inception_3a/output_0_split_0 (32, 256, 28, 28)
inception_3a/output_inception_3a/output_0_split_1 (32, 256, 28, 28)
inception_3a/output_inception_3a/output_0_split_2 (32, 256, 28, 28)
inception_3a/output_inception_3a/output_0_split_3 (32, 256, 28, 28)
inception_3b/1x1 (32, 128, 28, 28)
inception_3b/3x3_reduce (32, 128, 28, 28)
inception_3b/3x3 (32, 192, 28, 28)
inception_3b/5x5_reduce (32, 32, 28, 28)
inception_3b/5x5 (32, 96, 28, 28)
inception_3b/pool (32, 256, 28, 28)
inception_3b/pool_proj (32, 64, 28, 28)
inception_3b/output (32, 480, 28, 28)
pool3/3x3_s2 (32, 480, 14, 14)
pool3/3x3_s2_pool3/3x3_s2_0_split_0 (32, 480, 14, 14)
pool3/3x3_s2_pool3/3x3_s2_0_split_1 (32, 480, 14, 14)
pool3/3x3_s2_pool3/3x3_s2_0_split_2 (32, 480, 14, 14)
pool3/3x3_s2_pool3/3x3_s2_0_split_3 (32, 480, 14, 14)
inception_4a/1x1 (32, 192, 14, 14)
inception_4a/3x3_reduce (32, 96, 14, 14)
inception_4a/3x3 (32, 208, 14, 14)
inception_4a/5x5_reduce (32, 16, 14, 14)
inception_4a/5x5 (32, 48, 14, 14)
inception_4a/pool (32, 480, 14, 14)
inception_4a/pool_proj (32, 64, 14, 14)
inception_4a/output (32, 512, 14, 14)
inception_4a/output_inception_4a/output_0_split_0 (32, 512, 14, 14)
inception_4a/output_inception_4a/output_0_split_1 (32, 512, 14, 14)
inception_4a/output_inception_4a/output_0_split_2 (32, 512, 14, 14)
inception_4a/output_inception_4a/output_0_split_3 (32, 512, 14, 14)
inception_4a/output_inception_4a/output_0_split_4 (32, 512, 14, 14)
loss1/ave_pool (32, 512, 4, 4)
loss1/conv (32, 128, 4, 4)
loss1/fc (32, 1024)
loss1/classifier (32, 1000)
loss1/loss1 ()
inception_4b/1x1 (32, 160, 14, 14)
inception_4b/3x3_reduce (32, 112, 14, 14)
inception_4b/3x3 (32, 224, 14, 14)
inception_4b/5x5_reduce (32, 24, 14, 14)
inception_4b/5x5 (32, 64, 14, 14)
inception_4b/pool (32, 512, 14, 14)
inception_4b/pool_proj (32, 64, 14, 14)
inception_4b/output (32, 512, 14, 14)
inception_4b/output_inception_4b/output_0_split_0 (32, 512, 14, 14)
inception_4b/output_inception_4b/output_0_split_1 (32, 512, 14, 14)
inception_4b/output_inception_4b/output_0_split_2 (32, 512, 14, 14)
inception_4b/output_inception_4b/output_0_split_3 (32, 512, 14, 14)
inception_4c/1x1 (32, 128, 14, 14)
inception_4c/3x3_reduce (32, 128, 14, 14)
inception_4c/3x3 (32, 256, 14, 14)
inception_4c/5x5_reduce (32, 24, 14, 14)
inception_4c/5x5 (32, 64, 14, 14)
inception_4c/pool (32, 512, 14, 14)
inception_4c/pool_proj (32, 64, 14, 14)
inception_4c/output (32, 512, 14, 14)
inception_4c/output_inception_4c/output_0_split_0 (32, 512, 14, 14)
inception_4c/output_inception_4c/output_0_split_1 (32, 512, 14, 14)
inception_4c/output_inception_4c/output_0_split_2 (32, 512, 14, 14)
inception_4c/output_inception_4c/output_0_split_3 (32, 512, 14, 14)
inception_4d/1x1 (32, 112, 14, 14)
inception_4d/3x3_reduce (32, 144, 14, 14)
inception_4d/3x3 (32, 288, 14, 14)
inception_4d/5x5_reduce (32, 32, 14, 14)
inception_4d/5x5 (32, 64, 14, 14)
inception_4d/pool (32, 512, 14, 14)
inception_4d/pool_proj (32, 64, 14, 14)
inception_4d/output (32, 528, 14, 14)
inception_4d/output_inception_4d/output_0_split_0 (32, 528, 14, 14)
inception_4d/output_inception_4d/output_0_split_1 (32, 528, 14, 14)
inception_4d/output_inception_4d/output_0_split_2 (32, 528, 14, 14)
inception_4d/output_inception_4d/output_0_split_3 (32, 528, 14, 14)
inception_4d/output_inception_4d/output_0_split_4 (32, 528, 14, 14)
loss2/ave_pool (32, 528, 4, 4)
loss2/conv (32, 128, 4, 4)
loss2/fc (32, 1024)
loss2/classifier (32, 1000)
loss2/loss2 ()
inception_4e/1x1 (32, 256, 14, 14)
inception_4e/3x3_reduce (32, 160, 14, 14)
inception_4e/5x5_reduce (32, 32, 14, 14)
inception_4e/3x3 (32, 320, 14, 14)
inception_4e/5x5 (32, 128, 14, 14)
inception_4e/pool (32, 528, 14, 14)
inception_4e/pool_proj (32, 128, 14, 14)
inception_4e/output (32, 832, 14, 14)
pool4/3x3_s2 (32, 832, 7, 7)
pool4/3x3_s2_pool4/3x3_s2_0_split_0 (32, 832, 7, 7)
pool4/3x3_s2_pool4/3x3_s2_0_split_1 (32, 832, 7, 7)
pool4/3x3_s2_pool4/3x3_s2_0_split_2 (32, 832, 7, 7)
pool4/3x3_s2_pool4/3x3_s2_0_split_3 (32, 832, 7, 7)
inception_5a/1x1 (32, 256, 7, 7)
inception_5a/3x3_reduce (32, 160, 7, 7)
inception_5a/3x3 (32, 320, 7, 7)
inception_5a/5x5_reduce (32, 32, 7, 7)
inception_5a/5x5 (32, 128, 7, 7)
inception_5a/pool (32, 832, 7, 7)
inception_5a/pool_proj (32, 128, 7, 7)
inception_5a/output (32, 832, 7, 7)
inception_5a/output_inception_5a/output_0_split_0 (32, 832, 7, 7)
inception_5a/output_inception_5a/output_0_split_1 (32, 832, 7, 7)
inception_5a/output_inception_5a/output_0_split_2 (32, 832, 7, 7)
inception_5a/output_inception_5a/output_0_split_3 (32, 832, 7, 7)
inception_5b/1x1 (32, 384, 7, 7)
inception_5b/3x3_reduce (32, 192, 7, 7)
inception_5b/3x3 (32, 384, 7, 7)
inception_5b/5x5_reduce (32, 48, 7, 7)
inception_5b/5x5 (32, 128, 7, 7)
inception_5b/pool (32, 832, 7, 7)
inception_5b/pool_proj (32, 128, 7, 7)
inception_5b/output (32, 1024, 7, 7)
pool5/7x7_s1 (32, 1024, 1, 1)
loss3/classifier (32, 1000)
loss3/loss3 ()

"""