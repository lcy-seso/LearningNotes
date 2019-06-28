# [How Much Attention Do you need](http://aclweb.org/anthology/P18-1167)

1. The performance of recurrent and convolutional models can be very close to the Transformer performance by borrowing concepts from the Transformer architecture, but not using self-attention.
1. Self-attention is much more important for the encoder side than for the decoder side.
    * In the encoder side, self-attention can be replaced by a RNN or CNN without a loss in performance in most settings.
    * One surpising experimental result is even a model without any target side self-attention performs well.
1. Source attention on lower encoder layers brings no additional benefit.
1. The largest gains come from multiple attention mechanisms and residual feed-forward layers.
