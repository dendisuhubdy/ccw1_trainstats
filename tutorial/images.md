# Visualizing Images with Visdom

For example you would want to look at your GAN pictures while training, at first your code looks like this

```
def save_images(images, num_x, num_y, out_file=None):
    if out_file is None:
        logger.warning('`out_file` not provided. Not saving.')
    else:
        if _options['quantized']:
            images = dequantize(images)

        elif _options['use_tanh']:
            images = 0.5 * (images + 1.)


        dim_c, dim_x, dim_y = images.shape[-3:]
        if dim_c == 1:
            plt.imsave(out_file,
                       (images.reshape(num_x, num_y, dim_x, dim_y)
                        .transpose(0, 2, 1, 3)
                        .reshape(num_x * dim_x, num_y * dim_y)),
                       cmap='gray') 
        else:
            scipy.misc.imsave(
                out_file, (images.reshape(num_x, num_y, dim_c, dim_x, dim_y)
                           .transpose(0, 3, 1, 4, 2)
                           .reshape(num_x * dim_x, num_y * dim_y, dim_c)))
```


The trick for Visualizing images in Visdom is to load it as an array then modify it to reduce the stream size and do a `vis.image` to 
send it to the Visdom server

```
import numpy as np
from PIL import Image
```

then we need to load it in array format using `np.array`

```
im = Image.fromarray(arr)
arr = np.array(im)
viz.image(arr, opts=dict(title=title, caption=caption), win='image_{}'.format(image_id), env=env)
```

It is that simple.

Sometimes the problems that we'd face is that the image is too big to stream and we would want to make the visualization tiled. For this case we could use the function `tile_raster_image` from [deeplearning.net](http://deeplearning.net/tutorial/code/utils.py).

```
def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

```

Finally with some modifications for image captions and labels, we'd arrive at this script

```

def save_images(images, num_x, num_y, env='main', out_file=None, labels=None,
                margin_x=5, margin_y=5, image_id=0, caption='', title=''):
    logger.info('Saving images')
    if labels is not None:
        if _options['is_caption']:
            margin_x = 80
            margin_y = 50
        elif _options['is_attribute']:
            margin_x = 25
            margin_y = 200
        elif _options['label_names'] is not None:
            margin_x = 20
            margin_y = 25
        else:
            margin_x = 5
            margin_y = 12

    if out_file is None:
        logger.warning('`out_file` not provided. Not saving.')
    else:

        if _options['quantized']:
            images = dequantize(images)
        elif _options['use_tanh']:
            images = 0.5 * (images + 1.)

        images = images * 255.

        dim_c, dim_x, dim_y = images.shape[-3:]

        if dim_c == 1:
            arr = tile_raster_images(
                X=images, img_shape=(dim_x, dim_y), tile_shape=(num_x, num_y),
                tile_spacing=(margin_y, margin_x))
            fill = 255
        else:
            arrs = []
            for c in xrange(dim_c):
                arr = tile_raster_images(
                    X=images[:, c].copy(), img_shape=(dim_x, dim_y),
                    tile_shape=(num_x, num_y),
                    tile_spacing=(margin_y, margin_x),
                    )
                arrs.append(arr)
            arr = np.array(arrs).transpose(1, 2, 0)
            fill = (255, 255, 255)

        im = Image.fromarray(arr)
        if labels is not None:
            try:
                font = ImageFont.truetype(
                    '/usr/share/fonts/truetype/freefont/FreeSans.ttf', 9)
            except:
                font = ImageFont.truetype(
                    '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf', 9)

            idr = ImageDraw.Draw(im)
            for i, label in enumerate(labels):
                x_ = (i % num_x) * (dim_x + margin_x)
                y_ = (i // num_x) * (dim_y + margin_y) + dim_y
                if _options['is_caption']:
                    l_ = ''.join([CHAR_MAP[j] for j in label])
                    if len(l_) > 20:
                        l_ = '\n'.join(
                            [l_[x:x+20] for x in range(0, len(l_), 20)])
                elif _options['is_attribute']:
                    attribs = [j for j, a in enumerate(label) if a == 1]
                    l_ = '\n'.join(_options['label_names'][a] for a in attribs)
                elif _options['label_names'] is not None:
                    l_ = _options['label_names'][label]
                    l_ = l_.replace('_', '\n')
                else:
                    l_ = str(label)
                idr.text((x_, y_), l_, fill=fill, font=font)
        arr = np.array(im)
        if arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)
        viz.image(arr, opts=dict(title=title, caption=caption),
                         win='image_{}'.format(image_id), env=env)
        im.save(out_file)
        logger.info('Done saving image')

```
