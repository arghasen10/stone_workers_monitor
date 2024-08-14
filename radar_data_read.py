import argparse
import csv
import math
import os
import time
from inspect import trace
from operator import add
from time import sleep
import glob
import matplotlib.pyplot as plt
import numpy as np
import serial
import sys

# import fft
from scipy.fftpack import fft
# smooth

os_name = os.environ.get("OS")
framePeriodicity = 0
configs = {
    "pointcloud": "Configurations/pointcloud_configuration.cfg",
    "macro": "Configurations/macro_5fps.cfg",
    "micro": "Configurations/micro_2fps.cfg",
}
# CLIport = {}
# Dataport = {}
byteBuffer = np.zeros(2**15, dtype="uint8")
byteBufferLength = 0
rangeAzimuthHeatMapGridInit = 0
xlin, ylin = [], []
NUM_ANGLE_BINS = 64
range_depth = 10
range_width = 5
changes_happening = 0
change_conf = False

header = [
    "Date",
    "Time",
    "numObj",
    "rangeIdx",
    "range",
    "dopplerIdx",
    "doppler",
    "peakVal",
    "x",
    "y",
    "z",
    "rp",
    "noiserp",
    "zi",
    "rangeDoppler",
    "rangeArray",
    "dopplerArray",
    # "interFrameProcessingTime",
    # "transmitOutputTime",
    # "interFrameProcessingMargin",
    # "interChirpProcessingMargin",
    # "activeFrameCPULoad",
    # "interFrameCPULoad",
]

#  use this if you want to create csv file  , comment the other one file_create ()
def file_create():
    filename = os.path.abspath("")
    filename += time.strftime("/%Y%m%d_%H%M%S")
    filename += ".csv"
    with open(filename, "w") as f:
        csv.DictWriter(f, fieldnames=header).writeheader()

    return filename




# //use the below one if you are making .pkl file  , comment the other one file_create ()
# def file_create():
#     filename = os.path.abspath("")
#     if os.name == "nt":  # For Windows
#         filename += time.strftime("\\%Y%m%d_%H%M%S")
#     elif os.name == "posix":  # For Linux, Unix, or macOS
#         filename += time.strftime("/%Y%m%d_%H%M%S")
#     filename += ".pkl"
#     with open(filename, "wb") as f:
#         pickle.dump({}, f)  # Dump an empty dictionary if needed
#     return filename


# ------------------------------------------------------------------


# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    global CLIport
    global Dataport

    CLIport = ""
    Dataport = ""
    # Open the serial ports for the configuration and the data ports

    # Raspberry pi
    ports = glob.glob('/dev/ttyACM*')
    print(ports)
    CLIport = serial.Serial('/dev/ttyACM0', 115200)
    Dataport = serial.Serial('/dev/ttyACM1', 921600)

    # elif os_name == "Windows_NT":
    #     CLIport = serial.Serial("COM6", 115200)
    #     Dataport = serial.Serial("COM9", 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip("\r\n") for line in open(configFileName)]
    for i in config:
        print(i)
        CLIport.write((i + "\n").encode())
        time.sleep(0.01)

    return CLIport, Dataport


# ------------------------------------------------------------------


# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    global framePeriodicity
    configParameters = (
        {}
    )  # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip("\r\n") for line in open(configFileName)]
    for i in config:
        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 2

        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1

            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2

            digOutSampleRate = int(splitWords[11])

        # Get the information about the frame configuration
        elif "frameCfg" in splitWords[0]:
            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = int(float(splitWords[5]))

    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
        2 * freqSlopeConst * 1e12 * numAdcSamples
    )
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
        2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"]
    )
    configParameters["dopplerResolutionMps"] = 3e8 / (
        2
        * startFreq
        * 1e9
        * (idleTime + rampEndTime)
        * 1e-6
        * configParameters["numDopplerBins"]
        * numTxAnt
    )
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (
        2 * freqSlopeConst * 1e3
    )
    configParameters["maxVelocity"] = 3e8 / (
        4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt
    )

    return configParameters


# ------------------------------------------------------------------

# Helper methods for processing


def tensor_f(vec1, vec2):
    t = []
    for r in range(0, len(vec1)):
        t.append(np.multiply(np.array(vec2), vec1[r]))
    return t


def meshgrid(xvec, yvec):
    x = []
    y = []
    for r in range(0, len(yvec)):
        for c in range(0, len(xvec)):
            x.append(xvec[c])
            y.append(yvec[r])
    return [x, y]


def reshape_rowbased(vec, rows, cols):
    t = []
    start = 0
    for r in range(0, rows):
        row = vec[start : start + cols]
        t.append(row)
        start += cols
    return t


def change_conf_callback():
    global CLIport, Dataport, configParameters, configFileName, byteBuffer, byteBufferLength
    byteBuffer = np.zeros(2**15, dtype="uint8")
    byteBufferLength = 0
    print(
        "############################ changing configuration to macro ##########################"
    )
    time.sleep(2)
    configFileName = "Configurations/macro_5fps.cfg"
    CLIport, Dataport = serialConfig(configFileName)
    configParameters = parseConfigFile(configFileName)


def processDetectedpoints(byteVec, vecIdx, numDetectedObjects, configParameters):
    x = np.zeros(numDetectedObjects,dtype=np.float32)
    y = np.zeros(numDetectedObjects,dtype=np.float32)
    z = np.zeros(numDetectedObjects,dtype=np.float32)
    velocity = np.zeros(numDetectedObjects,dtype=np.float32)

    
    for objectNum in range(numDetectedObjects):
        startidX = vecIdx+objectNum*16      # size of object is 16
        x[objectNum] = byteVec[startidX:startidX + 4].view(dtype=np.float32)
        startidX += 4
        y[objectNum] = byteVec[startidX:startidX + 4].view(dtype=np.float32)
        startidX += 4
        z[objectNum] = byteVec[startidX:startidX + 4].view(dtype=np.float32)
        startidX += 4
        velocity[objectNum] = byteVec[startidX:startidX + 4].view(dtype=np.float32)
        startidX += 4
    
    range_val = np.sqrt(x**2+y**2+z**2)
    rangeidX = np.floor(range_val/configParameters["rangeIdxToMeters"])
    doppleridX = np.floor(velocity/configParameters["dopplerResolutionMps"])
    doppler_Val = doppleridX * configParameters["dopplerResolutionMps"]


    # # Store the data in the detObj dictionary
    # detObj = {"numObj": numDetectedObjects, "x": x, "y": y, "z": z, "velocity":velocity, "rangeidX": rangeidX, "doppleridx": doppleridX, "range_val": range_val}
    # Store the data in the detObj dictionary
    detObj = {
        "numObj": numDetectedObjects,
        "rangeIdx": list(rangeidX),
        "range": list(range_val),
        "dopplerIdx": list(doppleridX),
        "doppler": list(doppler_Val),
        "x": list(x),
        "y": list(y),
        "z": list(z),
    }
    return detObj


def processRangeNoiseProfile(byteBuffer, idX, configParameters, isRangeProfile):
    traceidX = 0
    if isRangeProfile:
        traceidX = 0
    else:
        traceidX = 2
    numrp = 2 * configParameters["numRangeBins"]
    rp = byteBuffer[idX : idX + numrp]

    rp = list(map(add, rp[0:numrp:2], list(map(lambda x: 256 * x, rp[1:numrp:2]))))
    rp_x = (
        np.array(range(configParameters["numRangeBins"]))
        * configParameters["rangeIdxToMeters"]
    )
    idX += numrp
    if traceidX == 0:
        noiseObj = {"rp": rp}
        return noiseObj
    elif traceidX == 2:
        noiseObj = {"noiserp": rp}
        return noiseObj


def transform_radix2(real, imag):
    # Initialization
    if len(real) != len(imag):
        raise ValueError("Mismatched lengths")
    n = len(real)
    if n == 1:  # Trivial transform
        return
    levels = -1
    for i in range(32):
        if 1 << i == n:
            levels = i  # Equal to log2(n)
    if levels == -1:
        raise ValueError("Length is not a power of 2")
    
    cos_table = np.cos(2 * np.pi * np.arange(n // 2) / n)
    sin_table = np.sin(2 * np.pi * np.arange(n // 2) / n)
    
    # Bit-reversed addressing permutation
    for i in range(n):
        j = reverse_bits(i, levels)
        if j > i:
            real[i], real[j] = real[j], real[i]
            imag[i], imag[j] = imag[j], imag[i]
    
    # Cooley-Tukey decimation-in-time radix-2 FFT
    size = 2
    while size <= n:
        halfsize = size // 2
        tablestep = n // size
        for i in range(0, n, size):
            for j in range(i, i + halfsize):
                k = (j - i) * tablestep
                tpre = real[j + halfsize] * cos_table[k] + imag[j + halfsize] * sin_table[k]
                tpim = -real[j + halfsize] * sin_table[k] + imag[j + halfsize] * cos_table[k]
                real[j + halfsize] = real[j] - tpre
                imag[j + halfsize] = imag[j] - tpim
                real[j] += tpre
                imag[j] += tpim
        size *= 2
    return real, imag

def reverse_bits(x, bits):
    y = 0
    for i in range(bits):
        y = (y << 1) | (x & 1)
        x >>= 1
    return y


def transform_bluestein(real, imag):
    # Find a power-of-2 convolution length m such that m >= n * 2 + 1
    if len(real) != len(imag):
        raise ValueError("Mismatched lengths")
    n = len(real)
    m = 1
    while m < n * 2 + 1:
        m *= 2

    # Trigonometric tables
    cos_table = np.cos(np.pi * (np.arange(n) ** 2 % (n * 2)) / n)
    sin_table = np.sin(np.pi * (np.arange(n) ** 2 % (n * 2)) / n)

    # Temporary vectors and preprocessing
    areal = np.zeros(m)
    aimag = np.zeros(m)
    areal[:n] = real * cos_table + imag * sin_table
    aimag[:n] = -real * sin_table + imag * cos_table

    breal = np.zeros(m)
    bimag = np.zeros(m)
    breal[0] = cos_table[0]
    bimag[0] = sin_table[0]
    for i in range(1, n):
        breal[i] = breal[m - i] = cos_table[i]
        bimag[i] = bimag[m - i] = sin_table[i]

    # Convolution
    creal, cimag = convolve_complex(areal, aimag, breal, bimag)

    # Postprocessing
    real[:n] = creal[:n] * cos_table + cimag[:n] * sin_table
    imag[:n] = -creal[:n] * sin_table + cimag[:n] * cos_table
    return real, imag


def convolve_complex(xreal, ximag, yreal, yimag):
    n = len(xreal)
    xreal = np.fft.fft(xreal)
    ximag = np.fft.fft(ximag)
    yreal = np.fft.fft(yreal)
    yimag = np.fft.fft(yimag)
    
    zreal = xreal * yreal - ximag * yimag
    zimag = xreal * yimag + ximag * yreal
    
    creal = np.fft.ifft(zreal).real
    cimag = np.fft.ifft(zimag).real
    return creal, cimag


def transform(real, imag):
    if len(real) != len(imag):
        raise ValueError("Mismatched lengths")
    
    n = len(real)
    if n == 0:
        return
    elif (n & (n - 1)) == 0:  # Is power of 2
        real, imag = transform_radix2(real, imag)
    else:  # More complicated algorithm for arbitrary sizes
        real, imag = transform_bluestein(real, imag)
    return real, imag


def processAzimuthHeatMap(byteBuffer, idX, configParameters):
    numTxAnt = 2
    numRxAnt = 4
    numBytes = numRxAnt * numTxAnt * configParameters["numRangeBins"] * 4
    q = byteBuffer[idX : idX + numBytes]
    idX += numBytes
    q_rows = numTxAnt * numRxAnt
    q_cols = configParameters["numRangeBins"]
    q_idx = 0
    QQ = []
    NUM_ANGLE_BINS = 64
    for i in range(0, q_cols):
        real = np.zeros(NUM_ANGLE_BINS)
        img = np.zeros(NUM_ANGLE_BINS)
        for j in range(0, q_rows):
            real[j] = q[q_idx + 1] * 256 + q[q_idx]
            if real[j] > 32767:
                real[j] = real[j] - 65536
            img[j] = q[q_idx + 3] * 256 + q[q_idx + 2]
            if img[j] > 32767:
                img[j] = img[j] - 65536
            q_idx = q_idx + 4
        real, img = transform(real, img)
        for ri in range(0, NUM_ANGLE_BINS):
            real[ri] = int(math.sqrt(real[ri] * real[ri] + img[ri] * img[ri]))

        QQ.append(
            [
                y
                for x in [
                    real[int(NUM_ANGLE_BINS / 2) :],
                    real[0 : int(NUM_ANGLE_BINS / 2)],
                ]
                for y in x
            ]
        )
    fliplrQQ = []
    for tmpr in range(0, len(QQ)):
        fliplrQQ.append(QQ[tmpr][1:].reverse())
    global rangeAzimuthHeatMapGridInit
    if rangeAzimuthHeatMapGridInit == 0:
        angles_rad = np.multiply(
            np.arange(-NUM_ANGLE_BINS / 2 + 1, NUM_ANGLE_BINS / 2, 1),
            2 / NUM_ANGLE_BINS,
        )
        theta = []
        for ang in angles_rad:
            theta.append(math.asin(ang))
        range_val = np.multiply(
            np.arange(0, configParameters["numRangeBins"], 1),
            configParameters["rangeIdxToMeters"],
        )
        sin_theta = []
        cos_theta = []
        for t in theta:
            sin_theta.append(math.sin(t))
            cos_theta.append(math.cos(t))
        posX = tensor_f(range_val, sin_theta)
        posY = tensor_f(range_val, cos_theta)

        global xlin, ylin
        xlin = np.arange(-range_width, range_width, 2 * range_width / 99)
        if len(xlin) < 100:
            xlin = np.append(xlin, range_width)
        ylin = np.arange(0, range_depth, 1.0 * range_depth / 99)
        if len(ylin) < 100:
            ylin = np.append(ylin, range_depth)

        xiyi = meshgrid(xlin, ylin)
        rangeAzimuthHeatMapGridInit = 1

    zi = fliplrQQ
    zi = reshape_rowbased(zi, len(ylin), len(xlin))
    heatObj = {"zi": zi}
    return heatObj


def processRangeDopplerHeatMap(byteBuffer, idX, configParameters):
    # Get the number of bytes to read
    numBytes = (
        int(configParameters["numDopplerBins"])
        * int(configParameters["numRangeBins"])
        * 2
    )
    # Convert the raw data to int16 array
    payload = byteBuffer[idX : idX + numBytes]
    idX += numBytes
    # rangeDoppler = math.add(
    #     math.subset(rangeDoppler, math.index(math.range(0, numBytes, 2))),
    #     math.multiply(math.subset(rangeDoppler, math.index(math.range(1, numBytes, 2))), 256)
    # );

    rangeDoppler = list(
        map(
            add,
            payload[0:numBytes:2],
            list(map(lambda x: 256 * x, payload[1:numBytes:2])),
        )
    )  # wrong implementation. Need to update the range doppler at range index

    # rangeDoppler = payload.view(dtype=np.int16)
    # Some frames have strange values, skip those frames
    # TO DO: Find why those strange frames happen
    # if np.max(rangeDoppler) > 10000:
    #     return 0

    # Convert the range doppler array to a matrix
    rangeDoppler = np.reshape(
        rangeDoppler,
        (int(configParameters["numDopplerBins"]), configParameters["numRangeBins"]),
        "F",
    )  # Fortran-like reshape
    rangeDoppler = np.append(
        rangeDoppler[int(len(rangeDoppler) / 2) :],
        rangeDoppler[: int(len(rangeDoppler) / 2)],
        axis=0,
    )

    dopplerM = []
    rangeDoppler_list = list(rangeDoppler)
    for e in rangeDoppler_list:
        dopplerM.append(list(e))

    #
    # # Generate the range and doppler arrays for the plot
    rangeArray = (
        np.array(range(configParameters["numRangeBins"]))
        * configParameters["rangeIdxToMeters"]
    )
    dopplerArray = np.multiply(
        np.arange(
            -configParameters["numDopplerBins"] / 2,
            configParameters["numDopplerBins"] / 2,
        ),
        configParameters["dopplerResolutionMps"],
    )  # This is dopplermps from js.
    dopplerObj = {
        "rangeDoppler": dopplerM,
        "rangeArray": list(rangeArray),
        "dopplerArray": list(dopplerArray),
    }
    return dopplerObj


def processStatistics(byteBuffer, idX):
    word = [1, 2**8, 2**16, 2**24]
    interFrameProcessingTime = np.matmul(byteBuffer[idX : idX + 4], word)
    idX += 4
    transmitOutputTime = np.matmul(byteBuffer[idX : idX + 4], word)
    idX += 4
    interFrameProcessingMargin = np.matmul(byteBuffer[idX : idX + 4], word)
    idX += 4
    interChirpProcessingMargin = np.matmul(byteBuffer[idX : idX + 4], word)
    idX += 4
    activeFrameCPULoad = np.matmul(byteBuffer[idX : idX + 4], word)
    idX += 4

    interFrameCPULoad = np.matmul(byteBuffer[idX : idX + 4], word)
    idX += 4

    statisticsObj = {
        "interFrameProcessingTime": interFrameProcessingTime,
        "transmitOutputTime": transmitOutputTime,
        "interFrameProcessingMargin": interFrameProcessingMargin,
        "interChirpProcessingMargin": interChirpProcessingMargin,
        "activeFrameCPULoad": activeFrameCPULoad,
        "interFrameCPULoad": interFrameCPULoad,
    }
    return statisticsObj


def buffer_flush(idX, byteBufferLength, totalPacketLen):
    if 0 < idX < byteBufferLength:
        shiftSize = totalPacketLen

        byteBuffer[: byteBufferLength - shiftSize] = byteBuffer[
            shiftSize:byteBufferLength
        ]
        byteBuffer[byteBufferLength - shiftSize :] = np.zeros(
            len(byteBuffer[byteBufferLength - shiftSize :]), dtype="uint8"
        )
        byteBufferLength = byteBufferLength - shiftSize

        # Check that there are no errors with the buffer length
        if byteBufferLength < 0:
            byteBufferLength = 0


def readAndParseData16xx(Dataport, configParameters, filename):
    global byteBuffer, byteBufferLength, framePeriodicity, changes_happening, change_conf, configFileName
    finalObj = {"Date": time.strftime("%d/%m/%Y"), "Time": time.strftime("%H%M%S")}
    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12
    BYTE_VEC_ACC_MAX_SIZE = 2**15
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1
    MMWDEMO_UART_MSG_RANGE_PROFILE = 2
    MMWDEMO_OUTPUT_MSG_NOISE_PROFILE = 3
    MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4
    MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
    MMWDEMO_OUTPUT_MSG_STATS = 6
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO = 7
    maxBufferSize = 2**15
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    # Initialize variables
    magicOK = 0  # Checks if magic number has been read
    dataOK = 0  # Checks if the data has been read correctly
    frameNumber = 0
    detObj = {}
    tlv_type = 0

    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype="uint8")
    byteCount = len(byteVec)
    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength : byteBufferLength + byteCount] = byteVec[
            :byteCount
        ]
        byteBufferLength = byteBufferLength + byteCount

    # Check that the buffer has some data
    if byteBufferLength > 16:
        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc : loc + 8]
            if np.all(check == magicWord):
                startIdx.append(loc)

        # Check that startIdx is not empty
        if startIdx:
            # Remove the data before the first start index
            if 0 < startIdx[0] < byteBufferLength:
                byteBuffer[: byteBufferLength - startIdx[0]] = byteBuffer[
                    startIdx[0] : byteBufferLength
                ]
                byteBuffer[byteBufferLength - startIdx[0] :] = np.zeros(
                    len(byteBuffer[byteBufferLength - startIdx[0] :]), dtype="uint8"
                )
                byteBufferLength = byteBufferLength - startIdx[0]

            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

            # word array to convert 4 bytes to a 32-bit number
            word = [1, 2**8, 2**16, 2**24]

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12 : 12 + 4], word)

            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32-bit number
        word = [1, 2**8, 2**16, 2**24]

        # Initialize the pointer index
        idX = 0
        detectedPoints_idx = -1
        rangeProfile_idX = -1
        noiseProfileidX = -1
        azimuth_idX = -1
        rangeDoppler_idX = -1
        # Read the header
        magicNumber = byteBuffer[idX : idX + 8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX : idX + 4], word), "x")
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX : idX + 4], word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX : idX + 4], word), "x")
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX : idX + 4], word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX : idX + 4], word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX : idX + 4], word)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX : idX + 4], word)
        idX += 4
        subFrameNumber = np.matmul(byteBuffer[idX : idX + 4], word)
        idX += 4
        # Read the TLV messages
        for tlvIdx in range(numTLVs):
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]
            # Check the header of the TLV message
            tlv_type = np.matmul(byteBuffer[idX : idX + 4], word)
            idX += 4
            tlv_length = np.matmul(byteBuffer[idX : idX + 4], word)
            idX += 4
            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
                detectedPoints_idx = idX
            elif tlv_type == MMWDEMO_UART_MSG_RANGE_PROFILE:
                rangeProfile_idX = idX
            elif tlv_type == MMWDEMO_OUTPUT_MSG_NOISE_PROFILE:
                noiseProfileidX = idX
            elif tlv_type == MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP:
                azimuth_idX = idX
            elif tlv_type == MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP:
                rangeDoppler_idX = idX
            elif tlv_type == MMWDEMO_OUTPUT_MSG_STATS:
                print(tlv_type)
            elif tlv_type == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO:
                sideInfo_idX = idX

            idX += tlv_length
        
        if detectedPoints_idx > -1:
            detObjRes = processDetectedpoints(byteBuffer, detectedPoints_idx, numDetectedObj, configParameters)   
            finalObj.update(detObjRes)
        if rangeProfile_idX > -1:
            noiseObj = processRangeNoiseProfile(byteBuffer, rangeProfile_idX, configParameters, True)
            finalObj.update(noiseObj)
        if noiseProfileidX > -1:
            noiseObj = processRangeNoiseProfile(byteBuffer, rangeProfile_idX, configParameters, False)
            finalObj.update(noiseObj)
        if azimuth_idX > -1:
            heatObj = processAzimuthHeatMap(byteBuffer, azimuth_idX, configParameters)
            finalObj.update(heatObj)
        if rangeDoppler_idX > -1:
            dopplerObj = processRangeDopplerHeatMap(byteBuffer, rangeDoppler_idX, configParameters)
            finalObj.update(dopplerObj)
        print(finalObj)

        with open(filename, "a") as f:
            writer = csv.DictWriter(f, header)
            writer.writerow(finalObj)
        if 0 < idX < byteBufferLength:
            shiftSize = totalPacketLen

            byteBuffer[: byteBufferLength - shiftSize] = byteBuffer[
                shiftSize:byteBufferLength
            ]
            byteBuffer[byteBufferLength - shiftSize :] = np.zeros(
                len(byteBuffer[byteBufferLength - shiftSize :]), dtype="uint8"
            )
            byteBufferLength = byteBufferLength - shiftSize

            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

    return dataOK, frameNumber, finalObj


def parseArg():
    parser = argparse.ArgumentParser(description="Change Configuration")
    parser.add_argument(
        "--conf",
        help="Select configuration file",
        default="pointcloud",
        choices=["pointcloud", "macro", "micro"],
    )
    args = parser.parse_args()
    return args


# -------------------------    MAIN   -----------------------------------------

# Configurate the serial port
if __name__ == "__main__":
    # args = parseArg()
    print("calling")
    configFileName = 'Configurations/iwr1843_al.cfg' #configs[args.conf]
    CLIport, Dataport = serialConfig(configFileName)
    # Get the configuration parameters from the configuration file
    configParameters = parseConfigFile(configFileName)
    # print(configParameters)

    # Main loop
    detObj = {}
    frameData = {}
    currentIndex = 0
    filename = file_create()

    linecounter = 0


    capture_time = 0

    # if len(sys.argv) > 1:
    #     capture_time = int(sys.argv[1])*60
    # else:
    #     capture_time = 5*60

    # start_time = time.time()

    while True:
        linecounter += 1
        if linecounter > 1000000000:
            linecounter = 0
            filename = file_create()

        try:
            dataOk, frameNumber, finalObj = readAndParseData16xx(
                Dataport, configParameters, filename
            )
            if dataOk:
                # Store the current frame into frameData
                print(finalObj)
                currentIndex += 1
            # if args.conf == "pointcloud":
            #     time.sleep(0.03)
            # elif args.conf == "macro":
            #     time.sleep(0.2)
            # else :
            #     time.sleep(0.5)


            # time.sleep(0.03)  # Sampling frequency of 30 Hz

        # Stop the program and close everything if Ctrl + c is pressed
        except KeyboardInterrupt:
            CLIport.write("sensorStop\n".encode())
            CLIport.close()
            Dataport.close()
            break