import graphviz

def plot_tree(tree, filename=None, title="Tree", features_name=None, class_names=None, is_regression=False):
    """
    Visualizza e opzionalmente salva l'albero decisionale rappresentato da un oggetto DecisionNode.
    Args:
        tree: oggetto DecisionNode (radice dell'albero)
        filename: se specificato, salva l'immagine con questo nome (es: 'tree.png')
        title: titolo del grafo
        features_name: lista dei nomi delle feature
        class_names: lista dei nomi delle classi (opzionale, solo per classificazione)
        is_regression: True se albero di regressione, False se classificazione
    """
    dot = graphviz.Digraph(comment=title, node_attr={
        'shape': 'box',
        'style': "filled, rounded",
        'color': "lightblue",
        'fontname': "helvetica"
    })

    def get_class_color(idx):
        # Genera un colore unico per ogni classe (HSV cycling)
        import colorsys
        hue = (idx * 0.618033988749895) % 1.0  # golden ratio for good distribution
        rgb = colorsys.hsv_to_rgb(hue, 0.5, 1)
        return '#%02x%02x%02x' % tuple(int(255*x) for x in rgb)

    def regression_color(val, vmin, vmax):
        # Sfumatura dal verde chiaro (min) al verde scuro (max)
        if vmax == vmin:
            ratio = 0.0
        else:
            ratio = (val - vmin) / (vmax - vmin)
        # Verde chiaro: (200,255,200), Verde scuro: (0,100,0)
        r = int(200 * (1 - ratio))
        g = int(255 * (1 - ratio) + 100 * ratio)
        b = int(200 * (1 - ratio))
        return '#%02x%02x%02x' % (r, g, b)

    # Per regressione: trova min/max prediction nelle foglie
    def get_leaf_preds(node, preds):
        if node.is_leaf:
            preds.append(node.prediction)
        else:
            if node.left: get_leaf_preds(node.left, preds)
            if node.right: get_leaf_preds(node.right, preds)

    vmin, vmax = None, None
    if is_regression:
        preds = []
        get_leaf_preds(tree, preds)
        if preds:
            vmin, vmax = min(preds), max(preds)
        else:
            vmin, vmax = 0, 1

    def recursive_graph(dot, node, myindex):
        if node.is_leaf:
            myindex += 1
            # Determina colore e label
            if is_regression:
                color = regression_color(node.prediction, vmin, vmax)
                label = 'Leaf\nsamples={samples}\nprediction={pred}'.format(
                    samples=node.leaf_samples,
                    pred=round(node.prediction, 3)
                )
            else:
                class_idx = int(round(node.prediction))
                color = get_class_color(class_idx)
                class_label = class_names[class_idx] if class_names and class_idx < len(class_names) else str(class_idx)
                label = 'Leaf\nsamples={samples}\nclass={class_name}'.format(
                    samples=node.leaf_samples,
                    class_name=class_label
                )
            dot.node(str(myindex), label, fillcolor=color)
            return myindex, myindex
        else:
            myindex += 1
            feature_name = features_name[node.feature_index] if features_name else str(node.feature_index)
            s = '{feature_name} <= {threshold}\nsamples={samples}\ndepth={depth}'.format(
                feature_name=feature_name,
                threshold=round(node.threshold, 3),
                samples=node.leaf_samples,
                depth=node.depth
            )
            dot.node(str(myindex), s)
            lastindex = myindex
            if node.left is not None:
                index, lastindex = recursive_graph(dot, node.left, lastindex)
                dot.edge(str(myindex), str(index), label="True")
            if node.right is not None:
                index, lastindex = recursive_graph(dot, node.right, lastindex)
                dot.edge(str(myindex), str(index), label="False")
            return myindex, lastindex

    recursive_graph(dot, tree, 0)
    if filename:
        dot.render(filename, format='png', cleanup=True)
    else:
        dot.view()
