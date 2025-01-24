from config import np


def cross_entropy_error(y, t):
    """
    Args:
        y: batch prediction   | [N,C]
            N: batch size
            C: number of class
        t: batch target       | [N,] or [N,C]
            [N,]: label index form
            [N,C]: one-hot form
    ---
    Returns:
        loss: batch loss (average)  | scalar
    """
    # data point 입력이라면 batch size 1을 추가해 batch form으로 일반화
    if y.ndim == 1:
        y = y.reshape(1, y.size)    # [1,C]
        t = t.reshape(1, t.size)    # [1,] or [1,C]
    batch_size = y.shape[0]

    # target이 one-hot form이라변 label index form으로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)    # [N,]
                                # keepdims=False이므로 아래의 loss 계산 시 indexing에서 오류 X

    losses = np.negative(np.log(y[np.arange(batch_size), t] + 1e-7))    # [N,]
    loss = np.sum(losses) / batch_size

    return loss


def mean_squared_error(y, t):
    """
    Args:
        y: batch prediction     | [N,]
        t: batch target         | [N,]
    ---
    Returns:
        loss: batch loss        | scalar
    """
    batch_size = y.shape[0]
    losses = (y - t) ** 2               # [N,]
    return np.sum(losses) / batch_size
