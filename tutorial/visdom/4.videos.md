# How to show a video using visdom

The [`video`](https://github.com/facebookresearch/visdom/blob/master/py/__init__.py#L459-L514) method that we are using here takes the following arguments:

```
def video(self, tensor=None, videofile=None, win=None, env=None, opts=None):
    """
    This function plays a video. It takes as input the filename of the video
    or a `LxCxHxW` tensor containing all the frames of the video. The function
    does not support any plot-specific `options`.
    """
```

## Configuring the layout of a video object

If we want to render a 4D tensor, we can pass a 4D numpy matrix to ‘tensor’ argument of viz.video() (viz is our instance) to create a temporary .ogv file in a THEO codec compression. In this case, for simplicity, we are just presenting a simple video that depicts a series of images that change color from black to white. The following code can be used to construct an empty structure for a sequence of frames that is filled at each timestamp of video array with numbers ranging from 0 to 255:

```
video = np.empty([FRAMES, WIDTH, LENGTH, 3], dtype=np.uint8)
for n in range(FRAMES):
    video[n, :, :, :].fill(n)
viz.video(tensor=video)
```
In above, FRAMES stands for the number of frames in a sequence of images with a size of LENGTH by WIDTH pixel. Note that since we have not provided any frame rate, the default value of 25 frames per second, substitutes as an argument. The number 3 in the dimensions corresponds to BGR coloring of each frame image. At the time of writing this tutorial, visdom only supports 4 dimension BGR array for the tensor argument. For instance, black and white binary images return an error from visdom program [asserting the input to be a 4D array](https://github.com/facebookresearch/visdom/blob/master/py/__init__.py#L474). An easy sidestep, would be to replicate the matrix in fourth axis, e.g., `video4D = np.repeat(video_bw[:, :, :, np.newaxis], 3, axis=3)`.
Note that the file extension .ogv, although supported in HTML5, is not always supported in all browsers. One browser that supports .ogv videos is Opera version 48.

## Playing a local video file

In order to stream a local file we can use the following line:

```
viz.video(videofile=address_string_to_your_file)
```
## Combining altogether
After initializing the visdom instance and checking that our connection to visdom server is being made, we will create a video using a local sample GIF image as the tensor argument. We calculate an average delay originally stored in GIF file. In case of a non-local file, by providing the url address of a sample video, we can make sure that the video file we want to play is downloaded and stored locally. Then, by passing the downloaded file address to video method of our visdom instance, the video becomes available through the visdom server.

```
from visdom import Visdom
import numpy as np
import os.path
from sys import platform as _platform
import getpass
from skimage.io import imread
from PIL import Image, ImageSequence

viz = Visdom()

assert (viz.check_connection())

# video demo:
try:
    videofile="sample.gif"
    # reading GIF file as numpy array
    video = np.flip(m = imread(videofile), axis=3) # RGB to BGR

    # computing an average fps
    img_gif = Image.open(videofile)
    durations = []
    for frame in ImageSequence.Iterator(img_gif):
        try:
            durations.append(frame.info['duration'])
        except KeyError:
            # Ignore if there was no duration, we will not count that frame.
            pass
    total_duration = sum(durations) # in miliseconds

    gif_fps_ave = video.shape[0] * 1000 / total_duration # average fps

    viz.video(
        tensor=video,
        opts=dict(fps=gif_fps_ave)
    )
    
    # video demo: download video from http://media.w3.org/2010/05/sintel/trailer.ogv
    video_url = 'http://media.w3.org/2010/05/sintel/trailer.ogv'
    # linux
    if _platform == "linux" or _platform == "linux2":
        videofile = '/home/%s/Downloads/trailer.ogv' % getpass.getuser()
    # MAC OS X
    elif _platform == "darwin":
        videofile = '/Users/%s/Downloads/trailer.ogv' % getpass.getuser()

    # download video
    urllib.request.urlretrieve(video_url, videofile)
    
    if os.path.isfile(videofile):
        viz.video(videofile=videofile)
except ImportError as packageError:
    print("The following required packages don't exist:")
    print(packageError)
```

The following sources has being used to write some parts of the above tutorial:  
alimony, gifduration, (2017), https://github.com/alimony/gifduration/blob/master/gifduration.py  
facebookresearch, visdom, (2017), https://github.com/facebookresearch/visdom/blob/master/example/demo.py  
