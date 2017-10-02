import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import numpy as np

from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
from torch.utils.model_zoo import load_url
from torchvis import util
from PIL import Image # so, this woorks better than skimage, as torchvision transforms work best with PIL and Tensor.
from torchvision import transforms

# adapted from <https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb>
def show_images(img_original, saliency, title):
    # convert from c01 to 01c
    print(saliency.min(), saliency.max(), saliency.mean(), saliency.std())
    saliency = saliency[::-1]  # to BGR
    saliency = saliency.transpose(1, 2, 0)
    
#     # put back std fixing.
#     saliency = saliency * np.array([ 0.229, 0.224, 0.225 ])
    
    # plot the original image and the three saliency map variants
    plt.figure(figsize=(10, 10), facecolor='w')
    plt.subplot(2, 2, 1)
    plt.title('input')
    plt.imshow(np.asarray(img_original))
    plt.subplot(2, 2, 2)
    plt.title('abs. saliency')
    plt.imshow(np.abs(saliency).max(axis=-1), cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('pos. saliency')
    plt.imshow((np.maximum(0, saliency) / saliency.max()))
    plt.subplot(2, 2, 4)
    plt.title('neg. saliency')
    plt.imshow((np.maximum(0, -saliency) / -saliency.min()))
    plt.suptitle(title)
    plt.show()

def main():

    # get the orignial VGG. from https://github.com/jcjohnson/pytorch-vgg
    sd = load_url('https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth')

    # see <https://github.com/jcjohnson/pytorch-vgg/issues/3>
    sd['classifier.0.weight'] = sd['classifier.1.weight']
    sd['classifier.0.bias'] = sd['classifier.1.bias']
    del sd['classifier.1.weight']
    del sd['classifier.1.bias']

    sd['classifier.3.weight'] = sd['classifier.4.weight']
    sd['classifier.3.bias'] = sd['classifier.4.bias']
    del sd['classifier.4.weight']
    del sd['classifier.4.bias']

    vgg16 = models.vgg16(pretrained=False)
    vgg16.load_state_dict(sd)
    vgg16.cuda()
    # this is important. turn off dropout.
    _ = vgg16.eval()

    vis_param_dict, reset_state, remove_handles = util.augment_module(vgg16)

    img_to_use = Image.open('./4334173592_145856d89b.jpg')
    print(img_to_use.size)

    transform_1 = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
    ])

    # since it's 0-255 range.
    transform_2 = transforms.Compose([
        transforms.ToTensor(),
        # convert RGB to BGR
        # from <https://github.com/mrzhu-cool/pix2pix-pytorch/blob/master/util.py>
        transforms.Lambda(lambda x: torch.index_select(x, 0, torch.LongTensor([2, 1, 0]))),
        transforms.Lambda(lambda x: x*255),
        transforms.Normalize(mean = [103.939, 116.779, 123.68],
                              std = [ 1, 1, 1 ]),
    ])

    img_to_use_cropped = transform_1(img_to_use)
    img_to_use_cropped_tensor = transform_2(img_to_use_cropped)[np.newaxis]  # add first column for batching

    img_to_use_cropped_tensor.min(), img_to_use_cropped_tensor.max()


    img_to_use_cropped  # this is same as Lasagne example, this outputs an image

    input_img = Parameter(img_to_use_cropped_tensor.cuda(), requires_grad=True)
    if input_img.grad is not None:
        input_img.grad.data.zero_()
    vgg16.zero_grad()
    # wrap input in Parameter, so that gradients will be computed.
    raw_score = vgg16(input_img)
    raw_score_numpy = raw_score.data.cpu().numpy()
    print(raw_score_numpy.shape, np.argmax(raw_score_numpy.ravel()))
    loss = raw_score.sum()
    print('loss', loss)
    # second time, there's no output anymore, due to lack of hook
    # I didn't call it, as maybe zero_grad may have some interaction with it. Not sure. Just for safety.
    # _ = alexnet(Variable(img_to_use_cropped_tensor.cuda()))

    # so, forward one time, and backward multiple times.
    vis_param_dict['layer'] = 'classifier.6'
    vis_param_dict['method'] = util.GradType.NAIVE
    # which one coresponds
    # this is the max one. I assume it's the correct one.
    # indeed, it's correct.
    # 55 is n01729977, corresponding to "green snake, grass snake".
    vis_param_dict['index'] = 55
    # alexnet gives 64, which is n01749939, or green mamba. not sure which one is correct.
    loss.backward(retain_variables=True)

    show_images(img_to_use_cropped, input_img.grad.data.cpu().numpy()[0], 'naive')

    # so, forward one time, and backward multiple times.
    vis_param_dict['method'] = util.GradType.GUIDED
    # alexnet gives 64, which is n01749939, or green mamba. not sure which one is correct.
    if input_img.grad is not None:
        input_img.grad.data.zero_()
    vgg16.zero_grad()
    loss.backward(retain_variables=True)
    show_images(img_to_use_cropped, input_img.grad.data.cpu().numpy()[0], 'guided')

    # so, forward one time, and backward multiple times.
    vis_param_dict['method'] = util.GradType.DECONV
    # alexnet gives 64, which is n01749939, or green mamba. not sure which one is correct.
    if input_img.grad is not None:
        input_img.grad.data.zero_()
    vgg16.zero_grad()
    loss.backward(retain_variables=True)
    show_images(img_to_use_cropped, input_img.grad.data.cpu().numpy()[0], 'deconv')

    # reset state
    reset_state()

    img_to_use = Image.open('./5595774449_b3f85b36ec.jpg')
    img_to_use_cropped = transform_1(img_to_use)
    img_to_use_cropped_tensor = transform_2(img_to_use_cropped)[np.newaxis]  # add first column for batching
    img_to_use_cropped



    input_img = Parameter(img_to_use_cropped_tensor.cuda(), requires_grad=True)
    if input_img.grad is not None:
        input_img.grad.data.zero_()
    vgg16.zero_grad()
    # wrap input in Parameter, so that gradients will be computed.
    raw_score = vgg16(input_img)
    raw_score_numpy = raw_score.data.cpu().numpy()
    print(raw_score_numpy.shape, np.argmax(raw_score_numpy.ravel()))
    loss = raw_score.sum()
    print('loss', loss)

    # so, forward one time, and backward multiple times.
    vis_param_dict['layer'] = 'classifier.6'
    vis_param_dict['method'] = util.GradType.NAIVE
    vis_param_dict['index'] = 56
    loss.backward(retain_variables=True)
    show_images(img_to_use_cropped, input_img.grad.data.cpu().numpy()[0], 'naive')


    # so, forward one time, and backward multiple times.
    vis_param_dict['method'] = util.GradType.GUIDED
    # alexnet gives 64, which is n01749939, or green mamba. not sure which one is correct.
    if input_img.grad is not None:
        input_img.grad.data.zero_()
    vgg16.zero_grad()
    loss.backward(retain_variables=True)
    show_images(img_to_use_cropped, input_img.grad.data.cpu().numpy()[0], 'guided')


    # then remove all callback
    reset_state()
    remove_handles()

if __name__=="__main__":
    main()
