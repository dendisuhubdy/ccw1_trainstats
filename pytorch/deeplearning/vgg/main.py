import torch
import io
import torchvision.models as models
import matplotlib
import matplotlib.pyplot as plt
import plotly.tools as tls

from visdom import Visdom
from PIL import Image

viz = Visdom(server='http://127.0.0.1', port = 51401)

def main():
    vgg = models.vgg16(pretrained=True)
    mm = vgg.double()
    filters = mm.modules
    body_model = [i for i in mm.children()][0]
    layer1 = body_model[0]
    tensor = layer1.weight.data.numpy()

    num_cols=6

    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols

    # starting the figure here

    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # plt.show()
    plt.title('VGG kernel 3x3')
    buf = io.BytesIO()
    #plotly_fig = tls.mpl_to_plotly(fig)
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)

    viz.heatmap(img, opts=dict(colormap='Greys'))

if __name__=="__main__":
    main()
