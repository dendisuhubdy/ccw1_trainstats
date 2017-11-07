# Visualizing Computational Graphs using Visdom

You can also visualize your computational graph from Theano on Visdom. Here's a step by step tutorial

## First step: do a theano.printing.debugprint on your loss/grad loss

For example when you define your loss

```
    print("Printing cost graph")
    cost_graph = theano.printing.debugprint(f_cost)
    display1 = viz.text(cost_graph)

    print("Printing grad function graph")
    grad_graph = theano.printing.debugprint(f_grad)
    display1 = viz.text(grad_graph)

    graph_cost_svg = theano.printing.pydotprint(f_cost ,  print_output_file=False, return_image=True, format='svg', scan_graphs=True,var_with_name_simple=True)
    graph_grad_svg = theano.printing.pydotprint(f_grad , print_output_file=False, return_image=True, format='svg', scan_graphs=True, var_with_name_simple=True)

```

you do a `theano.printing.pydotpring(var, return_image=True, format='svg')`

then you can send the SVG object using `viz.SVG`

```

    viz.svg(
            svgstr=str(graph_cost_svg),
            opts=dict(title='Theano Computational Graph - Cost'),
            )

    viz.svg(
            svgstr=str(graph_grad_svg),
            opts=dict(title='Theano Computational Graph - Cost Grad'),
            )

```

you can see the full example [here](https://github.com/dendisuhubdy/ccw1_trainstats/blob/master/theano/non-gans/lstm.py)
