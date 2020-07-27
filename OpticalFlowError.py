
import cv2
import numpy as np
import os
import sys

import Filesystem

def load_optical_flow(fn):
    return np.load(fn).astype(np.float32)

def split_image_filenames(opFn, prefix=None):
    '''
    opFn (string): The filename of the optical flow file.
    preFix (string): The prefix for the image file. Use None to use the prefix of opFn.

    opFn should be in the following format: 
    <leading path>/<zero leading 6-digit integer>_<zero leading 6-digit integer>_flow.npy.

    The filenames of the images are
    <zero leading 6-digit integer>.png
    '''

    parts = Filesystem.get_filename_parts(opFn)

    ss = parts[1].split('_')

    assert( len(ss) > 2 )

    if ( prefix is not None ):
        imgFn0 = os.path.join( prefix, '%s.png' % ( ss[0] ) )
        imgFn1 = os.path.join( prefix, '%s.png' % ( ss[1] ) )
    else:
        imgFn0 = os.path.join( parts[0], '%s.png' % ( ss[0] ) )
        imgFn1 = os.path.join( parts[0], '%s.png' % ( ss[1] ) )

    return imgFn0, imgFn1

def load_image(fn):
    return cv2.imread( fn, cv2.IMREAD_UNCHANGED )

def load_images( fns ):
    '''
    fns ( list of strings ): A list of filenames.
    '''

    imgs = []

    for f in fns:
        if ( not os.path.isfile(f) ):
            raise Exception('%s does not exist. ' % (f))

        img = load_image(f)
        imgs.append(img)

    return imgs

def convert_flow_2_maps(flow):
    '''
    flow (NumPy array): A two-channel NumPy 2D array. (H, W, C).

    Values in flow are recording the pixel movements from img0 to img1.
    '''

    H, W = flow.shape[:2]

    rangeX = np.arange(0, W)
    rangeY = np.arange(0, H)

    mapX, mapY = np.meshgrid( rangeX, rangeY )
    mapX = mapX.astype(np.float32)
    mapY = mapY.astype(np.float32)

    mapX = mapX + flow[ :, :, 0 ]
    mapY = mapY + flow[ :, :, 1 ]

    return mapX, mapY

def warp_by_flow(img, flow):
    '''
    img (NumPy array): A image loaded by cv2. Possible 2 channels.
    flow (NumPy array): A 2-channel NumPy array.

    img is the img1 of the image sequence. That means the second image
    of the temporal image sequence.
    '''

    mapX, mapY = convert_flow_2_maps(flow)

    warped = cv2.remap( img, mapX, mapY, cv2.INTER_LINEAR )

    # Mask computation.
    H, W = flow.shape[:2]
    maskX = np.logical_and( mapX >= 0, mapX < W )
    maskY = np.logical_and( mapY >= 0, mapY < H )
    mask  = np.logical_and( maskX, maskY )

    return warped, mask

def write_logical_image(fn, img):
    '''
    fn (string): Filename.
    img (NumPy array): A logical 2D array.
    '''

    img = img.astype(np.uint8)*255

    cv2.imwrite(fn, img, [ cv2.IMWRITE_PNG_COMPRESSION, 5 ] )

def diff_masked(imgs, mask):
    '''
    imgs (list of NumPy arrays): The two images.
    mask (NumPy logical array): A 2D logical array. The error will be evaluated
        on the True masked pixels only.
    '''

    H, W = imgs[0].shape[:2]

    img0 = imgs[0].reshape( (H, W, -1) ).astype(np.float32)
    img1 = imgs[1].reshape( (H, W, -1) ).astype(np.float32)

    img0 = img0.reshape( (-1, img0.shape[2]) )
    img1 = img1.reshape( (-1, img1.shape[2]) )

    mask = mask.reshape( (-1, ) )

    diff = np.zeros_like( img0 )

    diff[ mask, : ] = img0[mask, :] - img1[mask, :]

    return diff, mask

def diff_no_mask(imgs):
    '''
    imgs (list of NumPy arrays): The two images.
    '''

    diff = imgs[0].astype(np.float32) - imgs[1].astype(np.float32)
    return diff

def compute_photometric_error( imgs, mask=None ):
    '''
    imgs (list of NumPy arrays): The two images.
    mask (NumPy logical array): A 2D logical array. The error will be evaluated
        on the True masked pixels only.
    '''

    if ( mask is not None ):
        diff, mask = diff_masked( imgs, mask )
        error = diff[mask, :]

        H, W = imgs[0].shape[:2]
        diff = diff.reshape( ( H, W, -1 ) )
    else:
        diff = diff_no_mask( imgs )
        error = diff
    
    # print('error.shape = {}'.format(error.shape))
    error = np.linalg.norm( error, axis=1 )
    # print('error.shape = {}'.format(error.shape))

    return diff, [ error.min(), error.max(), error.mean() ]

def normalize_by_limits(x, limits):
    assert( limits[0] < limits[1] )

    x = np.clip( x, limits[0], limits[1] )
    x = ( x - limits[0] ) / ( limits[1] - limits[0] )

    return x

def self_normalize(x):
    minX = x.min()
    maxX = x.max()

    x = ( x - minX ) / ( maxX - minX )
    return x

def write_float_image(fn, img, limits=None):
    # Normalization.
    if ( limits is not None ):
        img = normalize_by_limits( img, limits )
    else:
        img = self_normalize(img)
    
    img = img * 255

    cv2.imwrite(fn, img, [ cv2.IMWRITE_PNG_COMPRESSION, 5 ])

def main():
    print('Hello, OpticalFlowError! ')

    # Read optical flow.
    ofFn = './SampleData2/000000_000001_flow.npy'

    flow = load_optical_flow(ofFn)
    imgFns = split_image_filenames(ofFn)

    # Read sample images.
    imgs = load_images(imgFns)

    # Warp.
    warped, mask = warp_by_flow(imgs[1], flow)

    # Save the warped image.
    cv2.imwrite( './SampleData2/Warped1.png', warped, [cv2.IMWRITE_PNG_COMPRESSION, 5] )

    # Save the mask as image.
    write_logical_image('./SampleData2/Mask.png', mask)

    # Compute the error.
    diff, errorStat = compute_photometric_error( [ imgs[0], warped ], mask )

    # print('diff.shape = {}'.format(diff.shape))
    print( 'Min %f, max %f, mean %f. ' % ( errorStat[0], errorStat[1], errorStat[2] ) )

    # Write the error map.
    write_float_image( './SampleData2/ErrorMap.png', np.linalg.norm(diff, axis=2) )

    return 0

if __name__ == '__main__':
    sys.exit( main() )