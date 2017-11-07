# Visualizing Matplotlib Objects on Visdom

For example we'd want to plot VGG on Visdom

```
import torch
import torchvision.models as models
import matplotlib
import matplotlib.pyplot as plt
import plotly.tools as tls


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

	plt.show()

if __name__=="__main__":
    main()

```

Here the problem is that rather want to display it on the Visdom server rather than doing a `plt` popup, so here we setup

```
from visdom import Visdom
viz = Visdom(server='http://127.0.0.1', port = 51401)
```

then below `plt.subplots_adjust` we would define

```
viz._send({
       data=plotly_fig.data,
       layout=plotly_fig.layout,
})

```

the full example could be seen [here](https://github.com/dendisuhubdy/ccw1_trainstats/blob/master/pytorch/vgg/main.py)
