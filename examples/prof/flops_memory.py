def forward(bs, depth, width, outdim):
    n_params = (depth - 1) * width ** 2 + width + outdim
    return bs * (n_params + (depth - 1) * width + outdim), bs * ((depth - 1) * width + outdim)


def backward(bs, depth, width, outdim):
    n_params = (depth - 1) * width ** 2 + width + outdim
    b_hidden = n_params - width ** 2 + outdim
    b_params = n_params
    return bs * (b_hidden + b_params), bs * width + n_params


def upd_params(bs, depth, width, outdim):
    n_params = (depth - 1) * width ** 2 + width + outdim
    return 2 * n_params, 0


def adam_metric(bs, depth, width, outdim):
    n_params = (depth - 1) * width ** 2 + width + outdim
    # square and sqrt
    return 2 * n_params, n_params


def adam_precond(bs, depth, width, outdim):
    n_params = (depth - 1) * width ** 2 + width + outdim
    # division
    return n_params, 0


def shampoo_metric(bs, depth, width, outdim):
    return (depth - 1) * 2 * width ** 3 + width * outdim * (width + outdim), \
           (depth - 1) * 2 * width ** 2 + width ** 2 + outdim ** 2


def shampoo_inverse(bs, depth, width, outdim):
    return (9 + 1/3) * ((depth - 1) * 2 * width ** 3 + width ** 3 + outdim ** 3), \
           (depth - 1) * 2 * width ** 2 + width ** 2 + outdim ** 2


def shampoo_precond(bs, depth, width, outdim):
    return (depth - 1) * 2 * width ** 3 + width * outdim * (width + outdim), 0


def kfac_metric(bs, depth, width, outdim):
    return bs * ((depth - 1) * 2 * width ** 2 + width ** 2 + outdim ** 2), \
           (depth - 1) * 2 * width ** 2 + width ** 2 + outdim ** 2


def kfac_inverse(bs, depth, width, outdim):
    return (depth - 1) * 2 * width ** 3 + width ** 3 + outdim ** 3, \
           (depth - 1) * 2 * width ** 2 + width ** 2 + outdim ** 2


def kfac_precond(bs, depth, width, outdim):
    return (depth - 1) * 2 * width ** 3 + width * outdim * (width + outdim), 0


def smw_ng_precond(bs, depth, width, outdim):
    gram = (depth - 1) * bs ** 2 * 2 * width + bs ** 2 * (width + outdim)
    return bs ** 3 / 3 + (3 + 2 * depth) * bs ** 2 + 2 * bs * depth + gram, bs ** 2 + bs


def lbfgs_precond(bs, depth, width, outdim, hist_size=20):
    n_params = (depth - 1) * width ** 2 + width + outdim
    return (10 * hist_size + 2) * n_params, 2 * hist_size * n_params + 2 * hist_size
