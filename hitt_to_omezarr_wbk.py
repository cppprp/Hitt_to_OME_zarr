import logging
import numpy as np
import os
import concurrent.futures as cf
import argparse
import shutil
import tifffile
import webknossos
from webknossos import SamplingModes
from webknossos import Mag
from webknossos.dataset.properties import VoxelSize, LengthUnit
import re
from pylibCZIrw import czi as pyczi
logging.basicConfig(format='%(asctime)s %(message)s')

#TODO: add memory watching
#TODO: ximg specific mask zeroing
def wipe_dir_and_contents(dirpath):
    # delete the dataset if already exists
    # wbknossos will throw an error if the 'layer' already exsits in the dataset
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
        print("Deleted")
def get_tiff16(f,n,minmin, maxmax):
    #mask_val = (-minmin*65535)/(minmin-maxmax)
    # conversion to 16 bit
    data[n,:,:]=(limit16(((tifffile.imread(f) - minmin) * 65535.0) / (maxmax - minmin)).astype(np.uint16))

def get_tiff8(f,n,minmin, maxmax):
    #conversion to 8 bit
    data[n,:,:]=(limit8(((tifffile.imread(f) - minmin) * 255.0) / (maxmax - minmin)).astype(np.uint8))

def get_tiff32(f,n, minmin, maxmax):
    # read 32 bit without conversion
    data[n, :, :] = tifffile.imread(f).astype(np.float32)
def get_tiffOR(f,n, minmin, maxmax):
    # will read in original data type
    data[n, :, :] = tifffile.imread(f)
def limit16(arr, mask_val):
    # limit values actually to the 16 bit range

    # arr[arr == mask_val] =0    <--- attempt  at clearing the mask
    arr[arr < 0] = 0
    arr[arr > 65535] = 65535
    return arr

def limit8(arr):
    # 8 bit limits
    arr[arr < 0] = 0
    arr[arr > 255] = 255
    return arr

def load_tif_stack(input_folder, start, end, _min, _max, _bit):
    # get tiff stack names
    fnames = sorted([f for f in os.listdir(input_folder) if not f.startswith('.')])
    # reset number of slices if none given
    if start is None:
        start = 0
    if end is None:
        end = len(fnames)
    # limit number of files
    fnames = fnames[start:end]
    logging.warning("Files to read: %s" % len(fnames))
    tiff_files = [os.path.join(input_folder, f) for f in fnames]

    # create data buffer with correct dimensions
    global data
    buffer = tifffile.TiffFile(tiff_files[0])
    rows = buffer.pages[0].shape[0]
    cols = buffer.pages[0].shape[1]
    data = np.zeros((len(tiff_files),rows,cols),dtype=_bit)

    # read data and convert bit depth if needed
    n = 0
    if _bit == "uint8":
        get_tiff = get_tiff8
    elif _bit == "uint16":
        get_tiff = get_tiff16
    elif _bit == "float32":
        get_tiff = get_tiff32
    else:
        get_tiff=get_tiffOR

    with cf.ThreadPoolExecutor(88, "treader_") as e:
        futures = []
        for f in tiff_files:
            futures.append(e.submit(get_tiff,f,n,_min,_max))
            n = n +1
        cf.wait(futures)

    logging.warning("Data loaded \n"
                "z: %s \n"
                "y: %s \n"
                "x: %s \n" % (data.shape[0], data.shape[1], data.shape[2]))
    print (np.max(data), data.dtype)
    return data

def processRaw(inputPath, start, end, _min, _max, _bit, endi):
    # This function will read a .raw buffer provided it ghas the dimensions in the name
    # eg bin4_Chicken_heart_D4_stitch_0.0_0.0_1384.0_914x1107x1892.raw
    # this is not beautiful, needs work
    dim = re.findall(r'\d+', inputPath)
    dimx=int(dim[-3])
    dimy=int(dim[-2])
    dimz=int(dim[-1])

    print("x ",dimx, " y ",dimy, " z ", dimz)
    # set pointer position if none given
    if start is None:
        start = 0
    if end is None:
        end = dimz
    # get endian order
    if endi == 'L':
        endi = '<f'
    else:
        endi ='>f'
    # read file into buffer
    global data
    f = open(inputPath,'rb')
    data = np.fromfile(f, dtype=endi, count=dimx * dimy * (end - start), offset=dimx * dimy * start * 4)
    # read data and convert bit depth if needed
    match _bit:
        case "uint8":
            data = (((data -_min) * 255.0) / (_max - _min))
            logging.warning("Converting data to 8 bit")
            data = limit8(data).astype(_bit)
        case "uint16":
            data = (((data -_min) * 65535.0) / (_max - _min))
            logging.warning("Converting data to 16 bit")
            data = limit16(data).astype(_bit)
        case _:
            logging.warning("Original data format used")
    # make into 3d array
    data = data.reshape(end-start, dimy, dimx)
    logging.warning("Data loaded \n"
                    "z: %s \n"
                    "y: %s \n"
                    "x: %s \n" % (data.shape[0], data.shape[1], data.shape[2]))
    print(np.max(data), data.dtype)
    return data
def read_czi(path):
    # this will read the czi files so far without any metadata, jsut images
    with pyczi.open_czi(path, pyczi.ReaderFileInputTypes.Standard) as czidoc:
        # based on the example we have 4 channels, but maybe need to account for others
        c0 = czidoc.read(plane={'C': 0})
        c1 = czidoc.read(plane={'C': 1})
        c2 = czidoc.read(plane={'C': 2})
        c3 = czidoc.read(plane={'C': 3})
        # assuming all channels have the same bit depth
        c0_pixel_type = czidoc.get_channel_pixel_type(0)
    image = np.stack((c0, c1, c2, c3))
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initialize ome-zarr parameters.')
    # Example input path:
    # "/mnt/ximg/2024/p3l-yschwab/RECON/20240414/RAW_DATA/TAL_5to20_20230627_NA_01_epo_01/recon_111_1"
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=False)
    parser.add_argument('--pxxy', type=int, required = True, default= 0.650)
    parser.add_argument('--pxz', type=int, required = False, default = None)
    parser.add_argument('--unit', type=str, default="micrometer")
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--start', type=int, required=False)
    parser.add_argument('--end', type=int, required=False)
    parser.add_argument('--min', type=float, required=False, default=None)
    parser.add_argument('--max', type=float, required=False, default=None)
    parser.add_argument('--bit', type=str, required=False, default="OR")
    parser.add_argument('--endian', type=str, required=False, default="L")
    parser.add_argument('--name', type=str, required=False)

    args = parser.parse_args()
    inputPath = args.input
    min_ = args.min
    max_ = args.max
    pxxy = args.pxxy
    pxz = args.pxz
    s = args.input.rsplit('/')
    if args.output is not None:
        savePath = args.output + '/' + s[-2] + '.ome.zarr/'
    else:
        savePath = args.input + '/' + s[-2] + '.ome.zarr/'
    # wipe output directory if already exists
    logging.warning("Writing to %s: " % savePath)
    wipe_dir_and_contents(savePath)
    # Get dtype for reading buffer
    match args.bit:
        case "8":
            bit = 'uint8'
        case "16":
            bit = 'uint16'
        case "32":
            bit = 'float32'
        case "OR":
            bit = None
    # dealing with stacks of tiff images
    if os.path.isdir(inputPath):
        if os.listdir(inputPath)[0].endswith('.tif') or os.listdir(inputPath)[0].endswith('.tiff'):
            data = load_tif_stack(inputPath, args.start, args.end, min_, max_, bit)
            data = np.swapaxes(data, 0, 2)
            # grabbing the datatype again just to be sure
            res_bit = data.dtype
            if (pxz is None):
            # convert data to zarr isotropic
                dataset = webknossos.Dataset(args.output, voxel_size_with_unit=VoxelSize((pxxy, pxxy, pxxy),
                                                                                         LengthUnit(args.unit)))
                layer = dataset.add_layer(layer_name=args.name, category="color", dtype_per_channel=res_bit, num_channels=1,
                                         data_format=webknossos.DataFormat.Zarr)
                mag1 = layer.add_mag("1", chunk_shape=(args.chunk, args.chunk, args.chunk))
                mag1.write(data=data)
                layer.downsample(coarsest_mag=Mag(8), sampling_mode=SamplingModes.ISOTROPIC)
            else:
            # convert anisotropic
                dataset = webknossos.Dataset(args.output, voxel_size_with_unit=VoxelSize((pxxy, pxxy, pxz),
                                                                                         LengthUnit(args.unit)))
                layer = dataset.add_layer(layer_name=args.name, category="color", dtype_per_channel=res_bit,
                                          num_channels=1,
                                          data_format=webknossos.DataFormat.Zarr)
                mag1 = layer.add_mag("1", chunk_shape=(args.chunk, args.chunk, args.chunk))
                mag1.write(data=data)
                layer.downsample(coarsest_mag=Mag(8), sampling_mode=SamplingModes.ANISOTROPIC)
        else:
            logging.warning("Input data format currently unsupported")

    elif os.path.isfile(inputPath):
        if inputPath.endswith('.raw'):
            # zarring the raw files, maybe not super important
            data = processRaw(inputPath, args.start, args.end, min_, max_, bit, args.endian)
            res_bit = data.dtype
            dataset = webknossos.Dataset(args.output, voxel_size_with_unit=VoxelSize((pxxy, pxxy, pxxy),
                                                                                     LengthUnit(args.unit)))
            layer = dataset.add_layer(layer_name=args.name, category="color", dtype_per_channel=res_bit, num_channels=1,
                                      data_format=webknossos.DataFormat.Zarr)
            mag1 = layer.add_mag("1", chunk_shape=(args.chunk, args.chunk, args.chunk))
            mag1.write(data=data)
            layer.downsample(coarsest_mag=Mag(8), sampling_mode=SamplingModes.ISOTROPIC)

        elif inputPath.endswith('.czi'):
            data = read_czi(inputPath)
            res_bit = data.dtype
            data = np.moveaxis(data, 2, 1)
            dataset = webknossos.Dataset(args.output,
                                         voxel_size_with_unit=VoxelSize((pxxy, pxxy, 1), LengthUnit(args.unit)))
            czi = dataset.add_layer(layer_name=args.name, category="color", dtype_per_channel=res_bit, num_channels=data.shape[0],
                                    data_format=webknossos.DataFormat.Zarr)
            mag1 = czi.add_mag("1", chunk_shape=(args.chunk, args.chunk, 1))
            mag1.write(data=data)
            # here probably dont need to downsample too much
            mag = Mag(4)
            czi.downsample(coarsest_mag=mag, sampling_mode=SamplingModes.CONSTANT_Z)

        else:
            logging.warning("Input data format currently unsupported")










