
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import Filesystem
import OpticalFlowError as ofe

def find_files(pattern):
    files = sorted( glob.glob(pattern, recursive=True) )

    if ( 0 == len(files) ):
        raise Exception('No files found by patter %s. ' % (pattern))

    return files

def deteremin_max_number(nFiles, nMax):
    if ( nMax > 0 ):
        if ( nMax < nFiles ):
            return nMax
    
    return nFiles

def process_flow(fn, imgPrefix):
    flow = ofe.load_optical_flow(fn)
    imgFns = ofe.split_image_filenames(fn, imgPrefix)

    # Read sample images.
    imgs = ofe.load_images(imgFns)

    # Warp.
    warped, mask = ofe.warp_by_flow(imgs[1], flow)

    # Compute the error.
    diff, errorStat = ofe.compute_photometric_error( [ imgs[0], warped ], mask )

    return errorStat

def handle_arguments():
    parser = argparse.ArgumentParser(description='Analysis the error pattrern of optical flow predictions. ')

    parser.add_argument('indir', type=str, 
        help='The input directory. ')

    parser.add_argument('pattern', type=str, 
        help='The search pattern for the flow file. ')

    parser.add_argument('imageprefix', type=str, 
        help='The the image folder prefix. ')

    parser.add_argument('outdir', type=str, 
        help='The working directory. ')

    parser.add_argument('--max-count', type=int, default=0, 
        help='Set the maximum number of flows to process. Debug use. Set 0 to disable.')

    args = parser.parse_args()

    return args

def main():
    print('Hello, LocalRun! ')

    args = handle_arguments()

    Filesystem.test_directory(args.outdir)

    # Find all the flow files.
    flows = find_files( '%s/**/%s' % ( args.indir, args.pattern ) )

    nFiles = deteremin_max_number( len(flows), args.max_count )

    errorStats = []

    imgPrefix = os.path.join( args.indir, args.imageprefix )

    for i in range(len(flows)):
        flowFn = flows[i]
        # print(flowFn)
        errorStat = process_flow( flowFn, imgPrefix )
        # print( 'Min %f, max %f, mean %f. ' % ( errorStat[0], errorStat[1], errorStat[2] ) )

        errorStats.append( errorStat )

    stat = np.array( errorStat, dtype=np.float32 )

    outFn = os.path.join( args.outdir, 'Stat.dat' )
    np.savetxt( outFn, stat, fmt='%+.12e' )       

    return 0

if __name__ == '__main__':
    sys.exit( main() )

