class SwinConfig:
    def __init__(self,
                 dim=None,
                 num_heads=None,
                 num_patch_x=None,
                 num_patch_y=None,
                 window_size=None,
                 shift_size=0,
                 num_mlp=None,
                 qkv_bias=True,
                 dropout_rate=None,
                 output_size=1):
        """

        :type num_patch_x: numeric
        """
        self.dim = dim  # number of input dimensions
        self.num_patch = (num_patch_x, num_patch_y)  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.output_size = output_size
