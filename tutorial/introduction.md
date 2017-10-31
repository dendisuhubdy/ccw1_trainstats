# Visdom - A Tool for Visualization

Visdom is a visualization tool that is developed by Facebook Artificial Intelligence Research (FAIR). This is a tutorial for using Visdom as a tool to do various visualizations for deep neural networks.

# Installation

To install Visdom, in your (Anaconda environment) just do

```
pip install visdom
```

# Usage

One Visdom is installed, now you can use use it for visualization. For a simple architectural understanding of visdom for the user, Visdom consists of a server and Python commands that you can use to stream your 
objects (experimental training statistics, GAN images, saliency maps) to it.

## Turning on the server

To turn on the server, you simply need to run

```
python -m visdom.server
```

This fires up the Visdom server at the default host and port which is localhost and port `8097`. If you would want to fire it up at another host for example if you host your Visdom server at your own public IP for teamwork purposes

```
python -m visdom.server --port <any_valid_TCP_port>
```


## Streaming to the server

Once you setup the server, now you can stream Visdom objects to it for example on your experiment script you can code this

```
from visdom import Visdom
viz = Visdom(server='http://www.visdomdomain.com', port=<previous_TCP_port>)
```

For example usually in my cases

```
from visdom import Visdom
viz = Visdom(server='http://suhubdy.com', port=51401)
```

would setup a Python object which refers to a TCP client stream to the TCP server server at `www.suhubdy.com` and at port `51401`. Some common mistakes that you might see if you stream to your server, for example

```
python - m visdom.server(server=www.suhubdy.com, port=51401)
```
this will NOT turn on your Visdom server, it needs to use the `http` protocol on top of it and the value of it must be between character literals ` ' ' ` .


