from ome_zarr.io import parse_url
import zarr
import logging
import numpy as np
import ome_zarr
import ome_zarr.scale
import ome_zarr.writer
import os
import concurrent.futures as cf
import argparse
import shutil
import tifffile

logging.basicConfig(format='%(asctime)s %(message)s')


def wipe_dir_and_contents(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
        print("Deleted")

def get_tiff(f,n,minmin, maxmax):
    #print(n,f)
    data[n,:,:]=(limit(((tifffile.imread(f) - minmin) * 65535.0) / (maxmax - minmin)).astype(np.uint16))

def limit(arr):
    # 16 bit limits
    arr[arr < 0] = 0
    arr[arr > 65535] = 65535
    return arr

def load_tif_stack(input_folder, start, end, _min, _max):
    fnames = sorted(os.listdir(input_folder))
    if start is None:
        start = 0
    if end is None:
        end = len(fnames)
    fnames = fnames[start:end]
    logging.warning("Files to read: %s" % len(fnames))

    tiff_files = [os.path.join(input_folder, f) for f in fnames]
    global data
    data = np.zeros((len(tiff_files),4096,4096),dtype=np.uint16)
    n = 0

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
    print(np.max(data))
    return data


def get_axes_and_trafos(mip, axis_names, units, resolution):
    logging.warning("Calculating transformations")
    axes = []
    for ax in axis_names:
        axis = {"name": ax, "type": "space"}
        unit = units.get(ax, None)
        if unit is not None:
            axis["unit"] = unit
        axes.append(axis)
    #TODO: there's the option to provide the scaling transform in Z but no scaler in the package
    is_scaled = {"z": False, "y": True, "x": True}
    trafos = [
        [{
            "scale": [resolution[ax] * 2 ** scale_level if is_scaled[ax] else resolution[ax] for ax in axis_names],
            "type": "scale"
        }]
        for scale_level in range(len(mip))
    ]

    return axes, trafos


def get_storage_opts(c):
    # default 64
    chunks = (c, c, c)
    return {"chunks": chunks}


def convert_image_data(data, group, resolution, units, c, name):
    scaler = ome_zarr.scale.Scaler()
    mip = scaler.local_mean(data)

    # specify the axis and transformation metadata
    axis_names = tuple("zyx")
    axes, trafos = get_axes_and_trafos(mip, axis_names, units, resolution)

    # provide additional storage options for zarr
    # the magic number can be bigger
    storage_opts = get_storage_opts(c)
    logging.warning("Writing ome-zarr")
    # write the data to ome.zarr
    ome_zarr.writer.write_multiscale(
        mip, group,
        axes=axes, coordinate_transformations=trafos,
        storage_options=storage_opts, name=name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initialize ome-zarr parameters.')
    # Example input path:
    # "/mnt/ximg/2024/p3l-yschwab/RECON/20240414/RAW_DATA/TAL_5to20_20230627_NA_01_epo_01/recon_111_1"
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=False)
    parser.add_argument('--px', type=float, default=0.650)
    parser.add_argument('--unit', type=str, default="micrometer")
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--start', type=int, required=False)
    parser.add_argument('--end', type=int, required=False)
    parser.add_argument('--min', type=float, required=False, default=None)
    parser.add_argument('--max', type=float, required=False, default=None)

    args = parser.parse_args()
    inputPath = args.input + '/tomo/'
    s = args.input.rsplit('/')
    if args.output is not None:
        savePath = args.output + '/' + s[-2] + '.ome.zarr/'
    else:
        savePath = args.input + '/' + s[-2] + '.ome.zarr/'

    logging.warning("Writing to %s: " %savePath)
    wipe_dir_and_contents(savePath)
    os.makedirs(savePath, exist_ok=True)

    resolution = {"z": args.px, "y": args.px, "x": args.px}
    units = {"z": args.unit, "y": args.unit, "x": args.unit}
    loc = parse_url(savePath, mode="w")
    chunk = args.chunk
    group = zarr.group(loc.store)
    data = load_tif_stack(inputPath, args.start, args.end, args.min, args.max)
    convert_image_data(data, group, resolution, units, chunk, name=s[-2])
    del data

