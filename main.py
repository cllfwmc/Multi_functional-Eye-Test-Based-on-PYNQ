import time
import random
from pynq import Overlay
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
import numpy as np
from pynq import Xlnk
import struct
from scipy.misc import imread
import cv2
import matplotlib.image as mpimg
from scipy import signal
from scipy.fftpack import fft
% matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
Wf_conv1=np.load('/home/xilinx/jupyter_notebooks/Final/Matrix/Wf1.npy')
bf_conv1=np.load('/home/xilinx/jupyter_notebooks/Final/Matrix/bf1.npy')
Wf_conv2=np.load('/home/xilinx/jupyter_notebooks/Final/Matrix/Wf2.npy')
bf_conv2=np.load('/home/xilinx/jupyter_notebooks/Final/Matrix/bf2.npy')
Wf_conv3=np.load('/home/xilinx/jupyter_notebooks/Final/Matrix/Wf3.npy')
bf_conv3=np.load('/home/xilinx/jupyter_notebooks/Final/Matrix/bf3.npy')

Mode = VideoMode(640,480,24)
hdmi_out = base.video.hdmi_out
hdmi_out.configure(Mode,PIXEL_BGR)
hdmi_out.start()
portx = "/dev/serial/by-id/usb-Texas_Instruments_XDS110__02.03.00.18__Embed_with_CMSIS-DAP_R0051028-if03"
bps = 921600
ser = serial.Serial(portx, bps, timeout = 0.2)


ol = Overlay("ai.bit")


def readbinfile(filename, size):
    f = open(filename, "rb")
    z = []
    for j in range(size):
        data = f.read(4)
        data_float = struct.unpack("f", data)[0]
        z.append(data_float)
    f.close()
    z = np.array(z)
    return z


def RunConv(conv, Kx, Ky, Sx, Sy, mode, relu_en, feature_in, W, bias, feature_out):
    conv.write(0x10, feature_in.shape[2]);
    conv.write(0x18, feature_in.shape[0]);
    conv.write(0x20, feature_in.shape[1]);
    conv.write(0x28, feature_out.shape[2]);
    conv.write(0x30, Kx);
    conv.write(0x38, Ky);
    conv.write(0x40, Sx);
    conv.write(0x48, Sy);
    conv.write(0x50, mode);
    conv.write(0x58, relu_en);
    conv.write(0x60, feature_in.physical_address);
    conv.write(0x68, W.physical_address);
    conv.write(0x70, bias.physical_address);
    conv.write(0x78, feature_out.physical_address);
    conv.write(0, (conv.read(0) & 0x80) | 0x01);
    tp = conv.read(0)
    while not ((tp >> 1) & 0x1):
        tp = conv.read(0);
    # print(tp);


def RunPool(pool, Kx, Ky, mode, feature_in, feature_out):
    pool.write(0x10, feature_in.shape[2]);
    pool.write(0x18, feature_in.shape[0]);
    pool.write(0x20, feature_in.shape[1]);
    pool.write(0x28, Kx);
    pool.write(0x30, Ky);
    pool.write(0x38, mode);
    pool.write(0x40, feature_in.physical_address);
    pool.write(0x48, feature_out.physical_address);
    pool.write(0, (pool.read(0) & 0x80) | 0x01);
    while not ((pool.read(0) >> 1) & 0x1):
        pass;


# 第一个卷积层参数
IN_WIDTH1 = 16
IN_HEIGHT1 = 16
IN_CH1 = 1

KERNEL_WIDTH1 = 3
KERNEL_HEIGHT1 = 3
X_STRIDE1 = 1
Y_STRIDE1 = 1

RELU_EN1 = 1
MODE1 = 1  # 0:VALID, 1:SAME
if (MODE1):
    X_PADDING1 = int((KERNEL_WIDTH1 - 1) / 2)
    Y_PADDING1 = int((KERNEL_HEIGHT1 - 1) / 2)
else:
    X_PADDING1 = 0
    Y_PADDING1 = 0

OUT_CH1 = 10
OUT_WIDTH1 = int((IN_WIDTH1 + 2 * X_PADDING1 - KERNEL_WIDTH1) / X_STRIDE1 + 1)
OUT_HEIGHT1 = int((IN_HEIGHT1 + 2 * Y_PADDING1 - KERNEL_HEIGHT1) / Y_STRIDE1 + 1)

# 第一个汇聚层参数

MODE11 = 2  # mode: 0:MEAN, 1:MIN, 2:MAX
IN_WIDTH11 = OUT_WIDTH1
IN_HEIGHT11 = OUT_HEIGHT1
IN_CH11 = OUT_CH1

KERNEL_WIDTH11 = 2
KERNEL_HEIGHT11 = 2

OUT_CH11 = IN_CH11
OUT_WIDTH11 = int(IN_WIDTH11 / KERNEL_WIDTH11)
OUT_HEIGHT11 = int(IN_HEIGHT11 / KERNEL_HEIGHT11)

# 第二个卷积层参数

IN_WIDTH2 = OUT_WIDTH11
IN_HEIGHT2 = OUT_HEIGHT11
IN_CH2 = OUT_CH11

KERNEL_WIDTH2 = 3
KERNEL_HEIGHT2 = 3
X_STRIDE2 = 1
Y_STRIDE2 = 1

RELU_EN2 = 1
MODE2 = 1  # 0:VALID, 1:SAME
if (MODE2):
    X_PADDING2 = int((KERNEL_WIDTH2 - 1) / 2)
    Y_PADDING2 = int((KERNEL_HEIGHT2 - 1) / 2)
else:
    X_PADDING2 = 0
    Y_PADDING2 = 0

OUT_CH2 = 20
OUT_WIDTH2 = int((IN_WIDTH2 + 2 * X_PADDING2 - KERNEL_WIDTH2) / X_STRIDE2 + 1)
OUT_HEIGHT2 = int((IN_HEIGHT2 + 2 * Y_PADDING2 - KERNEL_HEIGHT2) / Y_STRIDE2 + 1)

# 第二层汇聚层参数

MODE21 = 2  # mode: 0:MEAN, 1:MIN, 2:MAX
IN_WIDTH21 = OUT_WIDTH2
IN_HEIGHT21 = OUT_HEIGHT2
IN_CH21 = OUT_CH2

KERNEL_WIDTH21 = 2
KERNEL_HEIGHT21 = 2

OUT_CH21 = IN_CH21
OUT_WIDTH21 = int(IN_WIDTH21 / KERNEL_WIDTH21)
OUT_HEIGHT21 = int(IN_HEIGHT21 / KERNEL_HEIGHT21)

# 第三个卷积层参数

IN_WIDTH3 = OUT_WIDTH21
IN_HEIGHT3 = OUT_HEIGHT21
IN_CH3 = OUT_CH21

KERNEL_WIDTH3 = 3
KERNEL_HEIGHT3 = 3
X_STRIDE3 = 1
Y_STRIDE3 = 1

RELU_EN3 = 1
MODE3 = 1  # 0:VALID, 1:SAME
if (MODE3):
    X_PADDING3 = int((KERNEL_WIDTH3 - 1) / 2)
    Y_PADDING3 = int((KERNEL_HEIGHT3 - 1) / 2)
else:
    X_PADDING3 = 0
    Y_PADDING3 = 0

OUT_CH3 = 100
OUT_WIDTH3 = int((IN_WIDTH3 + 2 * X_PADDING3 - KERNEL_WIDTH3) / X_STRIDE3 + 1)
OUT_HEIGHT3 = int((IN_HEIGHT3 + 2 * Y_PADDING3 - KERNEL_HEIGHT3) / Y_STRIDE3 + 1)

# 第三层汇聚层参数

MODE31 = 2  # mode: 0:MEAN, 1:MIN, 2:MAX
IN_WIDTH31 = OUT_WIDTH3
IN_HEIGHT31 = OUT_HEIGHT3
IN_CH31 = OUT_CH3

KERNEL_WIDTH31 = 2
KERNEL_HEIGHT31 = 2

OUT_CH31 = IN_CH31
OUT_WIDTH31 = int(IN_WIDTH31 / KERNEL_WIDTH31)
OUT_HEIGHT31 = int(IN_HEIGHT31 / KERNEL_HEIGHT31)

# 第四个卷积层参数

IN_WIDTH4 = OUT_WIDTH31
IN_HEIGHT4 = OUT_HEIGHT31
IN_CH4 = OUT_CH31

KERNEL_WIDTH4 = 3
KERNEL_HEIGHT4 = 3
X_STRIDE4 = 1
Y_STRIDE4 = 1

RELU_EN4 = 1
MODE4 = 1  # 0:VALID, 1:SAME
if (MODE4):
    X_PADDING4 = int((KERNEL_WIDTH4 - 1) / 2)
    Y_PADDING4 = int((KERNEL_HEIGHT4 - 1) / 2)
else:
    X_PADDING4 = 0
    Y_PADDING4 = 0

OUT_CH4 = 100
OUT_WIDTH4 = int((IN_WIDTH4 + 2 * X_PADDING4 - KERNEL_WIDTH4) / X_STRIDE4 + 1)
OUT_HEIGHT4 = int((IN_HEIGHT4 + 2 * Y_PADDING4 - KERNEL_HEIGHT4) / Y_STRIDE4 + 1)

# 第四层汇聚层参数

MODE41 = 2  # mode: 0:MEAN, 1:MIN, 2:MAX
IN_WIDTH41 = OUT_WIDTH4
IN_HEIGHT41 = OUT_HEIGHT4
IN_CH41 = OUT_CH4

KERNEL_WIDTH41 = 2
KERNEL_HEIGHT41 = 2

OUT_CH41 = IN_CH41
OUT_WIDTH41 = int(IN_WIDTH41 / KERNEL_WIDTH41)
OUT_HEIGHT41 = int(IN_HEIGHT41 / KERNEL_HEIGHT41)

# 第五个卷积层参数

IN_WIDTH5 = OUT_WIDTH41
IN_HEIGHT5 = OUT_HEIGHT41
IN_CH5 = OUT_CH41

KERNEL_WIDTH5 = 1
KERNEL_HEIGHT5 = 1
X_STRIDE5 = 1
Y_STRIDE5 = 1

RELU_EN5 = 1
MODE5 = 0  # 0:VALID, 1:SAME
if (MODE5):
    X_PADDING5 = int((KERNEL_WIDTH5 - 1) / 2)
    Y_PADDING5 = int((KERNEL_HEIGHT5 - 1) / 2)
else:
    X_PADDING5 = 0
    Y_PADDING5 = 0

OUT_CH5 = 120
OUT_WIDTH5 = 1
OUT_HEIGHT5 = 1
print(OUT_WIDTH5)
print(OUT_HEIGHT5)

# 第六个卷积层参数

IN_WIDTH6 = 1
IN_HEIGHT6 = 1
IN_CH6 = 120

KERNEL_WIDTH6 = 1
KERNEL_HEIGHT6 = 1
X_STRIDE6 = 1
Y_STRIDE6 = 1

RELU_EN6 = 0
MODE6 = 0  # 0:VALID, 1:SAME
if (MODE6):
    X_PADDING6 = int((KERNEL_WIDTH6 - 1) / 2)
    Y_PADDING6 = int((KERNEL_HEIGHT6 - 1) / 2)
else:
    X_PADDING6 = 0
    Y_PADDING6 = 0

OUT_CH6 = 4
OUT_WIDTH6 = 1
OUT_HEIGHT6 = 1

xlnk = Xlnk();

ol.ip_dict
ol.download()
conv = ol.Conv_0
pool = ol.Pool_0
print("Overlay download finish");

# input image
image = xlnk.cma_array(shape=(IN_HEIGHT1, IN_WIDTH1, IN_CH1), cacheable=0, dtype=np.float32)

# conv1
W_conv1 = xlnk.cma_array(shape=(KERNEL_HEIGHT1, KERNEL_WIDTH1, IN_CH1, OUT_CH1), cacheable=0, dtype=np.float32)
b_conv1 = xlnk.cma_array(shape=(OUT_CH1), cacheable=0, dtype=np.float32)
h_conv1 = xlnk.cma_array(shape=(OUT_HEIGHT1, OUT_WIDTH1, OUT_CH1), cacheable=0, dtype=np.float32)
h_pool1 = xlnk.cma_array(shape=(OUT_HEIGHT11, OUT_WIDTH11, OUT_CH11), cacheable=0, dtype=np.float32)

W_conv2 = xlnk.cma_array(shape=(KERNEL_HEIGHT2, KERNEL_WIDTH2, IN_CH2, OUT_CH2), cacheable=0, dtype=np.float32)
b_conv2 = xlnk.cma_array(shape=(OUT_CH2), cacheable=0, dtype=np.float32)
h_conv2 = xlnk.cma_array(shape=(OUT_HEIGHT2, OUT_WIDTH2, OUT_CH2), cacheable=0, dtype=np.float32)
h_pool2 = xlnk.cma_array(shape=(OUT_HEIGHT21, OUT_WIDTH21, OUT_CH21), cacheable=0, dtype=np.float32)

W_conv3 = xlnk.cma_array(shape=(KERNEL_HEIGHT3, KERNEL_WIDTH3, IN_CH3, OUT_CH3), cacheable=0, dtype=np.float32)
b_conv3 = xlnk.cma_array(shape=(OUT_CH3), cacheable=0, dtype=np.float32)
h_conv3 = xlnk.cma_array(shape=(OUT_HEIGHT3, OUT_WIDTH3, OUT_CH3), cacheable=0, dtype=np.float32)
h_pool3 = xlnk.cma_array(shape=(OUT_HEIGHT31, OUT_WIDTH31, OUT_CH31), cacheable=0, dtype=np.float32)

W_conv4 = xlnk.cma_array(shape=(KERNEL_HEIGHT4, KERNEL_WIDTH4, IN_CH4, OUT_CH4), cacheable=0, dtype=np.float32)
b_conv4 = xlnk.cma_array(shape=(OUT_CH4), cacheable=0, dtype=np.float32)
h_conv4 = xlnk.cma_array(shape=(OUT_HEIGHT4, OUT_WIDTH4, OUT_CH4), cacheable=0, dtype=np.float32)
h_pool4 = xlnk.cma_array(shape=(OUT_HEIGHT41, OUT_WIDTH41, OUT_CH41), cacheable=0, dtype=np.float32)

W_conv5 = xlnk.cma_array(shape=(KERNEL_HEIGHT5, KERNEL_WIDTH5, IN_CH5, OUT_CH5), cacheable=0, dtype=np.float32)
b_conv5 = xlnk.cma_array(shape=(OUT_CH5), cacheable=0, dtype=np.float32)
h_conv5 = xlnk.cma_array(shape=(OUT_HEIGHT5, OUT_WIDTH5, OUT_CH5), cacheable=0, dtype=np.float32)

W_conv6 = xlnk.cma_array(shape=(KERNEL_HEIGHT6, KERNEL_WIDTH6, IN_CH6, OUT_CH6), cacheable=0, dtype=np.float32)
b_conv6 = xlnk.cma_array(shape=(OUT_CH6), cacheable=0, dtype=np.float32)
h_conv6 = xlnk.cma_array(shape=(OUT_HEIGHT6, OUT_WIDTH6, OUT_CH6), cacheable=0, dtype=np.float32)

w_conv1 = readbinfile("./W_conv1.bin", KERNEL_HEIGHT1 * KERNEL_WIDTH1 * IN_CH1 * OUT_CH1)
w_conv1 = w_conv1.reshape((KERNEL_HEIGHT1, KERNEL_WIDTH1, IN_CH1, OUT_CH1))
for i in range(KERNEL_HEIGHT1):
    for j in range(KERNEL_WIDTH1):

        for k in range(IN_CH1):
            for l in range(OUT_CH1):
                W_conv1[i][j][k][l] = w_conv1[i][j][k][l]
B_conv1 = readbinfile("./b_conv1.bin", OUT_CH1)
for i in range(OUT_CH1):
    b_conv1[i] = B_conv1[i]

w_conv2 = readbinfile("./W_conv2.bin", KERNEL_HEIGHT2 * KERNEL_WIDTH2 * IN_CH2 * OUT_CH2)
w_conv2 = w_conv2.reshape((KERNEL_HEIGHT2, KERNEL_WIDTH2, IN_CH2, OUT_CH2))
for i in range(KERNEL_HEIGHT2):
    for j in range(KERNEL_WIDTH2):
        for k in range(IN_CH2):
            for l in range(OUT_CH2):
                W_conv2[i][j][k][l] = w_conv2[i][j][k][l]
B_conv2 = readbinfile("./b_conv2.bin", OUT_CH2)
for i in range(OUT_CH2):
    b_conv2[i] = B_conv2[i]

w_conv3 = readbinfile("./W_conv3.bin", KERNEL_HEIGHT3 * KERNEL_WIDTH3 * IN_CH3 * OUT_CH3)
w_conv3 = w_conv3.reshape((KERNEL_HEIGHT3, KERNEL_WIDTH3, IN_CH3, OUT_CH3))
for i in range(KERNEL_HEIGHT3):

    for j in range(KERNEL_WIDTH3):

        for k in range(IN_CH3):
            for l in range(OUT_CH3):
                W_conv3[i][j][k][l] = w_conv3[i][j][k][l]
B_conv3 = readbinfile("./b_conv3.bin", OUT_CH3)
for i in range(OUT_CH3):
    b_conv3[i] = B_conv3[i]

w_conv4 = readbinfile("./W_conv4.bin", KERNEL_HEIGHT4 * KERNEL_WIDTH4 * IN_CH4 * OUT_CH4)
w_conv4 = w_conv4.reshape((KERNEL_HEIGHT4, KERNEL_WIDTH4, IN_CH4, OUT_CH4))
for i in range(KERNEL_HEIGHT4):
    for j in range(KERNEL_WIDTH4):
        for k in range(IN_CH4):
            for l in range(OUT_CH4):
                W_conv4[i][j][k][l] = w_conv4[i][j][k][l]
B_conv4 = readbinfile("./b_conv4.bin", OUT_CH4)
for i in range(OUT_CH4):
    b_conv4[i] = B_conv4[i]

w_conv5 = readbinfile("./W_conv5.bin", KERNEL_HEIGHT5 * KERNEL_WIDTH5 * IN_CH5 * OUT_CH5)
w_conv5 = w_conv5.reshape((KERNEL_HEIGHT5, KERNEL_WIDTH5, IN_CH5, OUT_CH5))
for i in range(KERNEL_HEIGHT5):
    for j in range(KERNEL_WIDTH5):
        for k in range(IN_CH5):
            for l in range(OUT_CH5):
                W_conv5[i][j][k][l] = w_conv5[i][j][k][l]
B_conv5 = readbinfile("./b_conv5.bin", OUT_CH5)
for i in range(OUT_CH5):
    b_conv5[i] = B_conv5[i]

w_conv6 = readbinfile("./W_conv6.bin", KERNEL_HEIGHT6 * KERNEL_WIDTH6 * IN_CH6 * OUT_CH6)
w_conv6 = w_conv6.reshape((KERNEL_HEIGHT6, KERNEL_WIDTH6, IN_CH6, OUT_CH6))
for i in range(KERNEL_HEIGHT6):
    for j in range(KERNEL_WIDTH6):
        for k in range(IN_CH6):
            for l in range(OUT_CH6):
                W_conv6[i][j][k][l] = w_conv6[i][j][k][l]
B_conv6 = readbinfile("./b_conv6.bin", OUT_CH6)
for i in range(OUT_CH6):
    b_conv6[i] = B_conv6[i]
WWW_conv1 = xlnk.cma_array(shape=(KERNEL_HEIGHT1, KERNEL_WIDTH1, IN_CH1, OUT_CH1), cacheable=0, dtype=np.float32)
bbb_conv1 = xlnk.cma_array(shape=(OUT_CH1), cacheable=0, dtype=np.float32)
hhh_conv1 = xlnk.cma_array(shape=(OUT_HEIGHT1, OUT_WIDTH1, OUT_CH1), cacheable=0, dtype=np.float32)
hhh_pool1 = xlnk.cma_array(shape=(OUT_HEIGHT11, OUT_WIDTH11, OUT_CH11), cacheable=0, dtype=np.float32)

WWW_conv2 = xlnk.cma_array(shape=(KERNEL_HEIGHT2, KERNEL_WIDTH2, IN_CH2, OUT_CH2), cacheable=0, dtype=np.float32)
bbb_conv2 = xlnk.cma_array(shape=(OUT_CH2), cacheable=0, dtype=np.float32)
hhh_conv2 = xlnk.cma_array(shape=(OUT_HEIGHT2, OUT_WIDTH2, OUT_CH2), cacheable=0, dtype=np.float32)
hhh_pool2 = xlnk.cma_array(shape=(OUT_HEIGHT21, OUT_WIDTH21, OUT_CH21), cacheable=0, dtype=np.float32)

WWW_conv3 = xlnk.cma_array(shape=(KERNEL_HEIGHT3, KERNEL_WIDTH3, IN_CH3, OUT_CH3), cacheable=0, dtype=np.float32)
bbb_conv3 = xlnk.cma_array(shape=(OUT_CH3), cacheable=0, dtype=np.float32)
hhh_conv3 = xlnk.cma_array(shape=(OUT_HEIGHT3, OUT_WIDTH3, OUT_CH3), cacheable=0, dtype=np.float32)
hhh_pool3 = xlnk.cma_array(shape=(OUT_HEIGHT31, OUT_WIDTH31, OUT_CH31), cacheable=0, dtype=np.float32)

WWW_conv4 = xlnk.cma_array(shape=(KERNEL_HEIGHT4, KERNEL_WIDTH4, IN_CH4, OUT_CH4), cacheable=0, dtype=np.float32)
bbb_conv4 = xlnk.cma_array(shape=(OUT_CH4), cacheable=0, dtype=np.float32)
hhh_conv4 = xlnk.cma_array(shape=(OUT_HEIGHT4, OUT_WIDTH4, OUT_CH4), cacheable=0, dtype=np.float32)
hhh_pool4 = xlnk.cma_array(shape=(OUT_HEIGHT41, OUT_WIDTH41, OUT_CH41), cacheable=0, dtype=np.float32)

WWW_conv5 = xlnk.cma_array(shape=(KERNEL_HEIGHT5, KERNEL_WIDTH5, IN_CH5, OUT_CH5), cacheable=0, dtype=np.float32)
bbb_conv5 = xlnk.cma_array(shape=(OUT_CH5), cacheable=0, dtype=np.float32)
hhh_conv5 = xlnk.cma_array(shape=(OUT_HEIGHT5, OUT_WIDTH5, OUT_CH5), cacheable=0, dtype=np.float32)

WWW_conv6 = xlnk.cma_array(shape=(KERNEL_HEIGHT6, KERNEL_WIDTH6, IN_CH6, OUT_CH6), cacheable=0, dtype=np.float32)
bbb_conv6 = xlnk.cma_array(shape=(OUT_CH6), cacheable=0, dtype=np.float32)
hhh_conv6 = xlnk.cma_array(shape=(OUT_HEIGHT6, OUT_WIDTH6, OUT_CH6), cacheable=0, dtype=np.float32)

www_conv1 = readbinfile("./WWW_conv1.bin", KERNEL_HEIGHT1 * KERNEL_WIDTH1 * IN_CH1 * OUT_CH1)
www_conv1 = www_conv1.reshape((KERNEL_HEIGHT1, KERNEL_WIDTH1, IN_CH1, OUT_CH1))
for i in range(KERNEL_HEIGHT1):
    for j in range(KERNEL_WIDTH1):

        for k in range(IN_CH1):
            for l in range(OUT_CH1):
                WWW_conv1[i][j][k][l] = www_conv1[i][j][k][l]
BBB_conv1 = readbinfile("./BBB_conv1.bin", OUT_CH1)
for i in range(OUT_CH1):
    bbb_conv1[i] = BBB_conv1[i]

www_conv2 = readbinfile("./WWW_conv2.bin", KERNEL_HEIGHT2 * KERNEL_WIDTH2 * IN_CH2 * OUT_CH2)
www_conv2 = www_conv2.reshape((KERNEL_HEIGHT2, KERNEL_WIDTH2, IN_CH2, OUT_CH2))
for i in range(KERNEL_HEIGHT2):
    for j in range(KERNEL_WIDTH2):
        for k in range(IN_CH2):
            for l in range(OUT_CH2):
                WWW_conv2[i][j][k][l] = www_conv2[i][j][k][l]
BBB_conv2 = readbinfile("./BBB_conv2.bin", OUT_CH2)
for i in range(OUT_CH2):
    bbb_conv2[i] = BBB_conv2[i]
/*
www_conv3 = readbinfile("./WWW_conv3.bin", KERNEL_HEIGHT3 * KERNEL_WIDTH3 * IN_CH3 * OUT_CH3)
www_conv3 = w_conv3.reshape((KERNEL_HEIGHT3, KERNEL_WIDTH3, IN_CH3, OUT_CH3))
for i in range(KERNEL_HEIGHT3):

    for j in range(KERNEL_WIDTH3):

        for k in range(IN_CH3):
            for l in range(OUT_CH3):
                WWW_conv3[i][j][k][l] = www_conv3[i][j][k][l]
BBB_conv3 = readbinfile("./BBB_conv3.bin", OUT_CH3)
for i in range(OUT_CH3):
    bbb_conv3[i] = BBB_conv3[i]

www_conv4 = readbinfile("./WWW_conv4.bin", KERNEL_HEIGHT4 * KERNEL_WIDTH4 * IN_CH4 * OUT_CH4)
www_conv4 =www_conv4.reshape((KERNEL_HEIGHT4, KERNEL_WIDTH4, IN_CH4, OUT_CH4))
for i in range(KERNEL_HEIGHT4):
    for j in range(KERNEL_WIDTH4):
        for k in range(IN_CH4):
            for l in range(OUT_CH4):
                WWW_conv4[i][j][k][l] = www_conv4[i][j][k][l]
BBB_conv4 = readbinfile("./BBB_conv4.bin", OUT_CH4)
for i in range(OUT_CH4):
    bbb_conv4[i] = BBB_conv4[i]

www_conv5 = readbinfile("./WWW_conv5.bin", KERNEL_HEIGHT5 * KERNEL_WIDTH5 * IN_CH5 * OUT_CH5)
www_conv5 = w_conv5.reshape((KERNEL_HEIGHT5, KERNEL_WIDTH5, IN_CH5, OUT_CH5))
for i in range(KERNEL_HEIGHT5):
    for j in range(KERNEL_WIDTH5):
        for k in range(IN_CH5):
            for l in range(OUT_CH5):
                WWW_conv5[i][j][k][l] = www_conv5[i][j][k][l]
BBB_conv5 = readbinfile("./BBB_conv5.bin", OUT_CH5)
for i in range(OUT_CH5):
    bbb_conv5[i] = BBB_conv5[i]

www_conv6 = readbinfile("./WWW_conv6.bin", KERNEL_HEIGHT6 * KERNEL_WIDTH6 * IN_CH6 * OUT_CH6)
www_conv6 = w_conv6.reshape((KERNEL_HEIGHT6, KERNEL_WIDTH6, IN_CH6, OUT_CH6))
for i in range(KERNEL_HEIGHT6):
    for j in range(KERNEL_WIDTH6):
        for k in range(IN_CH6):
            for l in range(OUT_CH6):
                WWW_conv6[i][j][k][l] = www_conv6[i][j][k][l]
BBB_conv6 = readbinfile("./BBB_conv6.bin", OUT_CH6)
for i in range(OUT_CH6):
    bbb_conv6[i] = BBB_conv6[i]

print("Finish initial")


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def choose(x):
    print(x)
    x = np.array(x)
    num = np.argmax(x)
    if num == 0:
        print('上')
    if num == 1:
        print('下')
    if num == 2:
        print('左')
    if num == 3:
        print('右')
    if num == 4:
        print('看不清')
    return num


def check_voice(ci):
    image = xlnk.cma_array(shape=(IN_HEIGHT1, IN_WIDTH1, IN_CH1), cacheable=0, dtype=np.float32)
    image1 = mpimg.imread("./T" + str(ci) + ".jpg")

    # rint(image1.shape)
    # image1=image1.reshape((IN_HEIGHT1,IN_WIDTH1,IN_CH1))
    for i in range(IN_HEIGHT1):
        for j in range(IN_WIDTH1):
            for k in range(IN_CH1):
                image[i][j][k] = (image1[i][j][k]) / 255
    print('run')
    RunConv(conv, KERNEL_WIDTH1, KERNEL_HEIGHT1, X_STRIDE1, Y_STRIDE1, MODE1, RELU_EN1, image, W_conv1, b_conv1,
            h_conv1)
    RunPool(pool, KERNEL_WIDTH11, KERNEL_HEIGHT11, MODE11, h_conv1, h_pool1)
    # conv2
    RunConv(conv, KERNEL_WIDTH2, KERNEL_HEIGHT2, X_STRIDE2, Y_STRIDE2, MODE2, RELU_EN2, h_pool1, W_conv2, b_conv2,
            h_conv2)
    RunPool(pool, KERNEL_WIDTH21, KERNEL_HEIGHT21, MODE21, h_conv2, h_pool2)

    RunConv(conv, KERNEL_WIDTH3, KERNEL_HEIGHT3, X_STRIDE3, Y_STRIDE3, MODE3, RELU_EN3, h_pool2, W_conv3, b_conv3,
            h_conv3)
    RunPool(pool, KERNEL_WIDTH31, KERNEL_HEIGHT31, MODE31, h_conv3, h_pool3)

    RunConv(conv, KERNEL_WIDTH4, KERNEL_HEIGHT4, X_STRIDE4, Y_STRIDE4, MODE4, RELU_EN4, h_pool3, W_conv4, b_conv4,
            h_conv4)
    RunPool(pool, KERNEL_WIDTH41, KERNEL_HEIGHT41, MODE41, h_conv4, h_pool4)

    RunConv(conv, KERNEL_WIDTH5, KERNEL_HEIGHT5, X_STRIDE5, Y_STRIDE5, MODE5, RELU_EN5, h_pool4, W_conv5, b_conv5,
            h_conv5)
    RunConv(conv, KERNEL_WIDTH6, KERNEL_HEIGHT6, X_STRIDE6, Y_STRIDE6, MODE6, RELU_EN6, h_conv5, W_conv6, b_conv6,
            h_conv6)
    return choose(softmax(h_conv6))

def check_voice1(ci):
    image = xlnk.cma_array(shape=(IN_HEIGHT1, IN_WIDTH1, IN_CH1), cacheable=0, dtype=np.float32)
    image1 = mpimg.imread("./T" + str(ci) + ".jpg")

    # rint(image1.shape)
    # image1=image1.reshape((IN_HEIGHT1,IN_WIDTH1,IN_CH1))
    for i in range(IN_HEIGHT1):
        for j in range(IN_WIDTH1):
            for k in range(IN_CH1):
                image[i][j][k] = (image1[i][j][k]) / 255
    print('run')
    RunConv(conv, KERNEL_WIDTH1, KERNEL_HEIGHT1, X_STRIDE1, Y_STRIDE1, MODE1, RELU_EN1, image, WWW_conv1, bbb_conv1,
            h_conv1)
    RunPool(pool, KERNEL_WIDTH11, KERNEL_HEIGHT11, MODE11, h_conv1, h_pool1)
    # conv2
    RunConv(conv, KERNEL_WIDTH2, KERNEL_HEIGHT2, X_STRIDE2, Y_STRIDE2, MODE2, RELU_EN2, h_pool1, WWW_conv2, bbb_conv2,
            h_conv2)
    RunPool(pool, KERNEL_WIDTH21, KERNEL_HEIGHT21, MODE21, h_conv2, h_pool2)

    RunConv(conv, KERNEL_WIDTH3, KERNEL_HEIGHT3, X_STRIDE3, Y_STRIDE3, MODE3, RELU_EN3, h_pool2, WWW_conv3, bbb_conv3,
            h_conv3)
    RunPool(pool, KERNEL_WIDTH31, KERNEL_HEIGHT31, MODE31, h_conv3, h_pool3)

    RunConv(conv, KERNEL_WIDTH4, KERNEL_HEIGHT4, X_STRIDE4, Y_STRIDE4, MODE4, RELU_EN4, h_pool3, WWW_conv4, bbb_conv4,
            h_conv4)
    RunPool(pool, KERNEL_WIDTH41, KERNEL_HEIGHT41, MODE41, h_conv4, h_pool4)

    RunConv(conv, KERNEL_WIDTH5, KERNEL_HEIGHT5, X_STRIDE5, Y_STRIDE5, MODE5, RELU_EN5, h_pool4, WWW_conv5, bbb_conv5,
            h_conv5)
    RunConv(conv, KERNEL_WIDTH6, KERNEL_HEIGHT6, X_STRIDE6, Y_STRIDE6, MODE6, RELU_EN6, h_conv5, WWW_conv6, bbb_conv6,
            h_conv6)
    return choose_3569(softmax(h_conv6))
def luyin(ci):
    pAudio = base.audio
    time.sleep(0.5)
    pAudio.record(1.2)

    pAudio.save('w.pdm')
    start = time.time()
    af_uint8 = np.unpackbits(pAudio.buffer.astype(np.int16)
                             .byteswap(True).view(np.uint8))
    end = time.time()
    start = time.time()
    af_dec = signal.decimate(af_uint8, 8, zero_phase=True)
    af_dec = signal.decimate(af_dec, 6, zero_phase=True)
    af_dec = signal.decimate(af_dec, 2, zero_phase=True)
    af_dec = (af_dec[10:-10] - af_dec[10:-10].mean())
    end = time.time()
    del af_uint8

    np.seterr(divide='ignore', invalid='ignore')
    matplotlib.style.use("classic")
    plt.figure(num=None, figsize=(0.8, 0.4), facecolor="white")
    plt.specgram(af_dec, Fs=32000)
    plt.axis('off');
    plt.savefig("./T" + str(ci) + ".jpg");

    img = Image.open("./T" + str(ci) + ".jpg");
    # 图片尺寸
    img_size = img.size
    h = img_size[1]  # 图片高度
    w = img_size[0]  # 图片宽度

    x1 = 10
    y1 = 4
    x2 = 74
    y2 = 36

    # 开始截取
    region = img.crop((x1, y1, x2, y2));
    # 保存图片
    region.save("./T" + str(ci) + ".jpg");

    plt.show()


def panduan(ci):
    
    conv = ol.Conv_0
    pool = ol.Pool_0
    pan = check_voice(ci)
    return pan

def panduan_3569(ci):
    
    conv = ol.Conv_0
    pool = ol.Pool_0
    pan = check_voice1(ci)
    return pan

def print_tu(tu):
    img = cv2.imread(tu)
    frame = cv2.resize(img[:][:],(640,480))
    outframe = hdmi_out.newframe()
    outframe[0:480,0:640,:] = frame[0:480,0:640,:]
    hdmi_out.writeframe(outframe)

def yuyin_tishi():
    out_wav("./Sound/be_1s.wav")

def print_E(distance,sl,fx):
    time.sleep(1)
    if(distance == 2):
        source = './Picture/E/2m/'
    else:
        source = './Picture/E/3m/'
    if(fx==0):
        print_tu(source+str(sl)+'/up.png')
    if(fx==1):
        print_tu(source+str(sl)+'/down.png')
    if(fx==2):
        print_tu(source+str(sl)+'/left.png')
    if(fx==3):
        print_tu(source+str(sl)+'/right.png')
  
def print_right_num(right_num):
    if(right_num == 1):
        out_wav('./Sound/correct.wav')
    else:
        out_wav('./Sound/wrong.wav')

def print_shili(shili):
    print_tu(('./Picture/Tip/'+str((int)(shili/10))+'.'+str(shili-(int)(shili/10)*10)+'.jpg'))
    out_wav("./Sound/tip5.wav")
    out_wav('./Sound/'+str((int)(shili/10))+'.'+str(shili-(int)(shili/10)*10)+'.wav')

def sound_model():
    time.sleep(1)
  
    out_key=1        
    min_num=40       
    max_num=53       
    now_num=46       
    menxian=1        
    all_num=1        
    last_num=0
    while(out_key):
        right_num=0    
        for i in range(all_num):
            fangxiang=random.randint(0,3)
            while(last_num==fangxiang):
                fangxiang=random.randint(0,3)
            last_num=fangxiang
            print(fangxiang)
            print_E(distance,now_num,fangxiang)
            yuyin_tishi()
            luyin(i)
            real_num=panduan(i)
            if(real_num==fangxiang):
                right_num+=1
        print_right_num(right_num)
        
        if(right_num>=menxian):
            min_num=now_num
            now_num=(int)((min_num+max_num)/2)
        if(right_num<menxian):
            max_num=now_num
            now_num=(int)((min_num+max_num)/2)
        if(min_num==max_num):
            print_shili(now_num)
            out_key=0


def choose_3569(x):
    x=x.reshape((4))
    x=np.array(x)
    num=np.argmax(x)
    if num==0:
        print('3')
        return 3
    if num==1:
        print('5')
        return 5
    if num==2:
        print('6')
        return 6
    if num==3:
        print('9')
        return 9


def distance_choose():
    index = 1
    print_tu('./Picture/Tip/distance0.jpg') 
    out_wav("./Sound/distance.wav")
    time.sleep(1)
    print_tu('./Picture/Tip/distance'+str(index)+'.jpg')
    while(1):
        flag = (int)(radar_detect())
        if((flag == 1) and (index != 1)):
            index = index - 1
            out_wav('./Sound/slide.wav')
            print_tu('./Picture/Tip/distance'+str(index)+'.jpg')
        if((flag == 1) and (index == 1)):
            index = 1
            print_tu('./Picture/Tip/distance'+str(index)+'.jpg')
        if((flag == 2) and (index != 2)):
            index = index + 1
            out_wav('./Sound/slide.wav')
            print_tu('./Picture/Tip/distance'+str(index)+'.jpg')
        if((flag == 2) and (index == 2)):
            index = 2
            print_tu('./Picture/Tip/distance'+str(index)+'.jpg')
        if( flag == 4):
            if(index == 1):
                out_wav('./Sound/confirm.wav')
                time.sleep(1)
                out_wav('./Sound/2m.wav')
                distance = 2  
                break
            if(index == 2):
                out_wav('./Sound/confirm.wav')
                time.sleep(1)
                out_wav('./Sound/3m.wav')
                distance = 3  
                break
        time.sleep(0.5)
    print_tu('./Picture/Tip/menu1.jpg')

def jinshi():
    index = 1
    print_tu('./Picture/Tip/moshi0.jpg') 
    out_wav("./Sound/moshi.wav")
    time.sleep(1)
    print_tu('./Picture/Tip/moshi'+str(index)+'.jpg')
    while(1):
        flag = (int)(radar_detect())
        if((flag == 1) and (index != 1)):
            index = index - 1
            out_wav('./Sound/slide.wav')
            print_tu('./Picture/Tip/moshi'+str(index)+'.jpg')
        if((flag == 1) and (index == 1)):
            index = 1
            print_tu('./Picture/Tip/moshi'+str(index)+'.jpg')
        if((flag == 2) and (index != 2)):
            index = index + 1
            out_wav('./Sound/slide.wav')
            print_tu('./Picture/Tip/moshi'+str(index)+'.jpg')
        if((flag == 2) and (index == 2)):
            index = 2
            print_tu('./Picture/Tip/moshi'+str(index)+'.jpg')
        if(flag == 4):
            if(index == 1):
                out_wav('./Sound/confirm.wav')
                time.sleep(1)
                radar_model()
                break
            if(index == 2):
                out_wav('./Sound/confirm.wav')
                time.sleep(1)
                sound_model()
                break
        time.sleep(0.5)
    print_tu('./Picture/Tip/menu2.jpg')
def semang():
    #print_tu('./Picture/Tip/introduction.jpg')
    #out_wav('./Sound/tip3.wav')
    time.sleep(1)
    list1 = [5,6]
    list2 = [3,9]
    xuantu1 = list1[random.randint(0,1)]
    xuantu2 = list2[random.randint(0,1)]
    print_tu('./Picture/semang/'+str(xuantu1)+'.jpg')
    time.sleep(2)
    yuyin_tishi()
    luyin(1)
    a = panduan_3569(1)
    if(a == xuantu1):
            outcome1 = 0  
    else:
        outcome1 = 1  
    print_tu('./Picture/semang/'+str(xuantu2)+'.jpg')
    time.sleep(2)
    yuyin_tishi()
    luyin(1)
    a = panduan_3569(1)
    if(a == xuantu2):
            outcome2 = 0  
    else:
        outcome2 = 1  

    if((outcome1 == 1)and(outcome2 == 0)):
        outcome = 1
        print_tu('./Picture/Tip/green_blind.jpg')
        out_wav('./Sound/green_blind.wav')
    if((outcome1 ==0)and(outcome2 ==1)):
        coutcome = 2
        print_tu('./Picture/Tip/red_blind.jpg')
        out_wav('./Sound/red_blind.wav')
    if((outcome1 == 1)and(outcome2 == 1)):
        coutcome = 3
        print_tu('./Picture/Tip/redgreen_blind.jpg')
        out_wav('./Sound/redgreen_blind.wav')
    if((outcome1 == 0)and(outcome2 == 0)):
        coutcome = 0
        print_tu('./Picture/Tip/normal.jpg')
        out_wav('./Sound/normal.wav')
    print_tu('./Picture/Tip/menu3.jpg')
def end():
    print_tu('./Picture/Tip/bye.jpg')
    out_wav('./Sound/bye.wav')
    hdmi_out.stop()
    #del hdmi_out
    
def flat_function(a):
    return np.reshape(a,360)

def relu(x):
    return np.maximum(0, x)

def choose_radar(x):
    x=x.reshape((4))
    x=np.array(x)
    num=np.argmax(x)
    if num==0:
        print('上')
    if num==1:
        print('下')
    if num==2:
        print('左')
    if num==3:
        print('右')
    return num+1

def radar_detect():
    ser = serial.Serial(portx, bps, timeout = 0.2)
    g1=np.zeros((8))
    g2=np.zeros((8))
    g3=np.zeros((8))
    g4=np.zeros((8))
    g5=np.zeros((8))
    g6=np.zeros((8))
    gg=0
    end=0
    dddd=8
    if (ser.isOpen()):
        print("open success")
        key=0
        ci=4
        while(ci):
            gesture=(ser.read(1))
            a=gesture[0]
            #print(a)
            if(a==2 and key==0):
                key=1
            if(a==1 and key==1):
                key=2
            if(a==4 and key==2):
                dddd-=1
                key=0
                pp=(ser.read(45+24))

                g=[]
                for i in range(6):
                    dd=pp[45+i*4]+256*pp[46+i*4]+65536*pp[47+i*4]+16777216*pp[48+i*4]
                    kk=0
                    if ((i == 0) or (i == 3 )or (i == 4 )or (i == 5)):
                        kk=(2147483645-dd)
                        if(kk>0):
                             kk=(kk-1000000000)/10000000
                        else:
                             kk=(kk+1000000000)/10000000
                    if(i==1):
                        kk=(dd-1220000000)/10000000
                    if (i == 2):
                        kk = (dd - 1080000000) / 1000000
                    g.append(kk)
                    if(i==0):
                        g1[gg]=kk
                    if (i == 1):

                        g2[gg]=kk
                        if(g2[gg-1]>4 and kk<g2[gg-1] and g2[gg-1]>g2[gg-2] and dddd<1):
                            end=1
                        if(end==1):
                            ci-=1
                    if (i ==2):
                        g3[gg]=kk
                    if (i == 3):
                        g4[gg]=kk
                    if(i==4):
                        g5[gg]=kk
                    if(i==5):
                        g6[gg]=kk
                    if(i==5):
                        gg+=1
                    if(gg==8):
                        gg=0
    else:
        print("open failed")

    ser.close()  # 关闭端口
    gg1=[]
    gg2=[]
    gg3=[]
    gg4=[]
    gg5=[]
    gg6=[]
    kg=gg-1
    for i in range(8):
        kg+=1
        if(kg==8):
            kg=0
        gg1.append(g1[kg])
    kg=gg-1
    for i in range(8):
        kg+=1
        if(kg==8):
            kg=0
        gg2.append(g2[kg])
    kg=gg-1
    for i in range(8):
        kg+=1
        if(kg==8):
            kg=0
        gg3.append(g3[kg])
    kg=gg-1
    for i in range(8):
        kg+=1
        if(kg==8):
            kg=0
        gg4.append(g4[kg])
    kg=gg-1
    for i in range(8):
        kg+=1
        if(kg==8):
            kg=0
        gg5.append(g5[kg])
    kg=gg-1
    for i in range(8):
        kg+=1
        if(kg==8):
            kg=0
        gg6.append(g6[kg])
    qq=[gg1,gg2,gg3,gg4,gg5,gg6]
    qq=np.array(qq)
    h_conv1=relu(np.matmul(qq,Wf_conv1))+bf_conv1
    h_conv2=relu(np.matmul(h_conv1,Wf_conv2))+bf_conv2
    h_flat=flat_function(h_conv2)
    prediction=softmax(np.matmul(h_flat,Wf_conv3)+bf_conv3)
    return choose_radar(prediction)

def radar_model1():
    out_key=1        
    min_num=40    
    max_num=53 
    now_num=46 
    menxian=1 
    all_num=1
    last_num=0
    while(out_key):
        right_num=0 
        for i in range(all_num):       
            fangxiang=random.randint(0,3)
            while(last_num==fangxiang):
                fangxiang=random.randint(0,3)
            last_num=fangxiang
            print(fangxiang)
            print_E(distance,now_num,fangxiang)
            real_num=radar_detect()
            out_wav("./Sound/be_1s.wav")
            time.sleep(1)
            if(real_num==fangxiang+1):
                right_num+=1
        print_right_num(right_num)
        if(right_num>=menxian):
            min_num=now_num
            now_num=(int)((min_num+max_num)/2)
        if(right_num<menxian):
            max_num=now_num
            now_num=(int)((min_num+max_num)/2)
        if(min_num==max_num):
            print_shili(now_num)
            out_key=0
    print_tu('./Picture/Tip/menu2.jpg') 
def menu_slide():
    index = 1
    print_tu('./Picture/Tip/menu0.jpg') 
    out_wav("./Sound/menu.wav")
    time.sleep(1)
    print_tu('./Picture/Tip/menu'+str(index)+'.jpg')
    while(1):
        flag = (int)(radar_detect())
        if((flag == 1) and (index != 1)):
            index = index - 1
            out_wav('./Sound/slide.wav')
            print_tu('./Picture/Tip/menu'+str(index)+'.jpg')
        if((flag == 1) and (index == 1)):
            index = 1
            
            print_tu('./Picture/Tip/menu'+str(index)+'.jpg')
        if((flag == 2) and (index != 4)):
            index = index + 1
            out_wav('./Sound/slide.wav')
            print_tu('./Picture/Tip/menu'+str(index)+'.jpg')
        if((flag == 2) and (index == 4)):
            index = 4
            
            print_tu('./Picture/Tip/menu'+str(index)+'.jpg')
        if( flag==4):
            if(index == 1):
                out_wav('./Sound/confirm.wav')
                distance_choose()
            if(index == 2):
                out_wav('./Sound/confirm.wav')
                jinshi()
            if(index == 3):
                out_wav('./Sound/confirm.wav')
                semang()
            if(index == 4):
                out_wav('./Sound/confirm.wav')
                end()
                break
        time.sleep(0.5)     
def radar_model():
    out_key=1 
    now_num=47
    last_num=0
    fx=0
    open_key=1
    fangxiang=random.randint(0,3)
    while(last_num==fangxiang):
        fangxiang=random.randint(0,3)
    last_num=fangxiang
    print(fangxiang)
    print_E(distance,now_num,fangxiang)
    real_num=radar_detect()
    out_wav("./Sound/be_1s.wav")
    time.sleep(1)
    if(real_num==fangxiang+1):
        print_right_num(1)
        fx=1
        now_num+=1
    else:
        print_right_num(0)
        now_num-=1
    while(open_key):
        fangxiang=random.randint(0,3)
        while(last_num==fangxiang):
            fangxiang=random.randint(0,3)
        last_num=fangxiang
        print(fangxiang)
        print_E(distance,now_num,fangxiang)
        real_num=radar_detect()
        out_wav("./Sound/be_1s.wav")
        time.sleep(1)
        if(real_num==fangxiang+1):
            print_right_num(1)
            if(fx==1):
                now_num+=1
                
            else:
                open_key=0
        else:
            print_right_num(0)
            if(fx==1):
                open_key=0
                now_num-=1
            else:
                now_num-=1
        if(now_num<40):
            now_num=40
            open_key=0
        if(now_num>53):
            now_num=53
            open_key=0
    print_shili(now_num)
    print_tu('./Picture/Tip/menu2.jpg')
def initial():
    print_tu('./Picture/Tip/daiji.jpg')
    out_wav('./Sound/daiji.wav')
    radar_detect()

def out_wav(path):
    pAudio = base.audio
    pAudio.load(path)
    pAudio.play()

distance = 2
initial()
menu_slide()    
