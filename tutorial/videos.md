# How to show a video using visdom

The `video` method that we are using here takes the following arguments:

```
def video(self, tensor=None, videofile=None, win=None, env=None, opts=None):
    """
    This function plays a video. It takes as input the filename of the video
    or a `LxCxHxW` tensor containing all the frames of the video. The function
    does not support any plot-specific `options`.
    """
```

## Configuring the layout of a video object

If we want to render a 4D tensor to play, we can pass a 4D numpy matrix to ‘tensor’ argument of viz.video() (viz is our instance) to create a temporary file in a THEO codec (a video compression format). In this case, for simplicity, we are just showing an empty video file. The following code can be used to construct an empty structure for a video object:

```
video = np.empty([256, 250, 250, 3], dtype=np.uint8)
for n in range(256):
    video[n, :, :, :].fill(n)
```


```
viz.video(tensor=video)
```

## Playing a local video file

In order to stream a local file we can use the following line:

```
viz.video(videofile='/Users/%s/Downloads/video_file.mp4' % getpass.getuser())
```
## Combining altogether
After initializing the visdom instance and checking that our connection is being made, we will create the above mentioned empty tensor video. Next, by providing the url address of a sample video we make sure that the file we want to play is downloaded and stored locally. Then, by passing the downloaded file address video to method of our visdom instance, the video becomes available through visdom server.

```
from visdom import Visdom
import numpy as np
import os.path
from sys import platform as _platform
import getpass

viz = Visdom()

assert (viz.check_connection())

# video demo:
try:

    video = np.empty([256, 250, 250, 3], dtype=np.uint8)
    for n in range(256):
        video[n, :, :, :].fill(n)
    viz.video(tensor=video)
    
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
```
