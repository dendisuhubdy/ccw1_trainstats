# Common Mistakes

When we do a `vis._send` we need to make send it as a `dict` instead, this is a common mistake that I've made during working with visdom

```
    plotly_fig = tls.mpl_to_plotly(fig)
    viz._send(
            data=plotly_fig.data,
            layout=plotly_fig.layout,
            )
```
