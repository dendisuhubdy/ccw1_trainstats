import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
import imageio
import logging


def load_mnist(dataset):
    data_dir = os.path.join("/data/lisatmp3/suhubdyd", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def visualize(images, sizei, margin_x = 5, margin_y = 5, image_id=0, caption='', title=''):
    # initialize visdom server
    import visdom as Visdom
    logger = logging.getlogger('GAN')
    Viz =Visdom(server='http://localhost', port=51401)
    logger.info('Streaming to visdom server')
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
    else:
        if _options['quantized']:
            images = dequantize(image)
        elif _options['use_tanh']:
            images = 0.5 * (images + 1.0)

            images = images *255
        dim_c, dim_x = image.shape[-3:]
        if dim_c == 1:
            arr = tile_raster_images(
                X=images, img_shape=(dim_x, dim_y), tile_shape=(num_x, num_y),
                tile_spacing=(margin_y, margin_x), bottom_margin=margin_y)
            fill = 255
        else:
            arrs = []
            for c in xrange(dim_c):
                arr = tile_raster_images(
                    X=images[:, c].copy(), img_shape=(dim_x, dim_y),
                    tile_shape=(num_x, num_y),
                    tile_spacing=(margin_y, margin_x),
                    bottom_margin=margin_y)
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



def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
