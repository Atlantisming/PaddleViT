import paddle
from paddle import nn
from paddle.nn import functional as F

from .former_config import MobileFormer_Config, get_head_hidden_size, MobileFormer_Config_Names


class Identify(nn.Layer):
    """Placeholder -- x = f(x)
    """
    def __init__(self):
        super(Identify, self).__init__(
                 name_scope="identify")

    def forward(self, inputs):
        x = inputs
        return x


class DepthWise_Conv(nn.Layer):
    """Deep convolution -- groups == in_channels
        Params Info:
            in_channels: Input channel number
            kernel_size: Convolution kernel size
            stride: Stride size
            padding: Padding size
        Forward Tips:
            - The number of input channels equals 
              the number of output channels
    """
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0):
        super(DepthWise_Conv, self).__init__(
                 name_scope="depthwise_conv")
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=in_channels,
                              groups=in_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        
    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Error: Please make sure the data of input has"+\
            "the same number of channel(in:{0}) with Conv Filter".format(
            inputs.shape[1])+"Config(set:{0})  in DepthWise_Conv.".format(
            self.in_channels)
        
        x = self.conv(inputs)

        return x


class PointWise_Conv(nn.Layer):
    """Point-by-point convolution -- 1x1
        Params Info:
            in_channels: Input channel number
            out_channels: Output channel number
            groups: 1x1 Convolution grouping number
        Forward Tips:
            - The input feature map equals the output feature map
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=1):
        super(PointWise_Conv, self).__init__(
                 name_scope="pointwise_conv")
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              groups=groups)
    
    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Error: Please make sure the data of input has"+\
            "the same number of channel(in:{0}) with Conv Filter".format(
            inputs.shape[1])+"Config(set:{0})  in PointWise_Conv.".format(
            self.in_channels)
        
        x = self.conv(inputs)

        return x


class MLP(nn.Layer):
    """Multi layer perceptron -- Two Linear Layers
        Params Info:
            in_features: Input feature size
            hidden_features: Hidden feature size
            out_features: Output feature size, default:None
            act: The activation function -- nn.Layer or nn.functional
            dropout_rate: Dropout rate
        Forward Tips:
            - The size of input features equals the size of output features
            - The last output is a linear output (not ACT activation)
    """
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features=None,
                 act=nn.GELU,
                 dropout_rate=0.):
        super(MLP, self).__init__(
                 name_scope="mlp")
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = in_features if out_features is None else out_features
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features,
                             out_features=self.out_features)
        self.act = act()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, inputs):
        assert inputs.shape[-1] == self.in_features, \
            "Error: Please make sure the data of input has"+\
            "the same number of features(in:{0}) with MLP Dense".format(
            inputs.shape[-1])+"Config(set:{0})  in MLP.".format(
            self.in_features)
        
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    """Attention module -- Support multi attention, can control QKV whether to use bias
        Params Info:
            embed_dims: Input feature size / embedding dimensions
            num_head: The number of Attention head
            qkv_bias: QKV mapping layer whether to use bias, default: True
            dropout_rate: Attentional outcome dropout rate
            attn_dropout_rate: Attentional distribution dropout rate
        Forward Tips:
            - The size of input features equals the size of output features
    """
    def __init__(self,
                 embed_dims,
                 num_head=1,
                 qkv_bias=True,
                 dropout_rate=0.,
                 attn_dropout_rate=0.):
        super(Attention, self).__init__(
                      name_scope="attn")
        self.embed_dims = embed_dims
        self.num_head = num_head
        self.head_dims = embed_dims // num_head
        self.scale = self.head_dims ** -0.5
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.qkv_proj = nn.Linear(in_features=embed_dims,
                                  out_features=3*self.num_head*self.head_dims,
                                  bias_attr=qkv_bias)
        self.out = nn.Linear(in_features=self.num_head*self.head_dims,
                             out_features=self.embed_dims)
        
        self.softmax = nn.Softmax()
        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)
    
    def transpose_qkv(self, qkv):
        assert len(qkv) == 3, \
            "Error: Please make sure the qkv_params has 3 items"+\
            ", but now it has {0} items in Attention-func:transpose_qkv.".format(
            len(qkv))

        q, k, v = qkv
        B, M, _ = q.shape
        q = q.reshape(shape=[B, M, self.num_head, self.head_dims]).transpose(
                      perm=[0, 2, 1, 3])
        k = k.reshape(shape=[B, M, self.num_head, self.head_dims]).transpose(
                      perm=[0, 2, 1, 3])
        v = v.reshape(shape=[B, M, self.num_head, self.head_dims]).transpose(
                      perm=[0, 2, 1, 3])
        return q, k, v

    def forward(self, inputs):
        assert inputs.shape[-1] == self.embed_dims, \
            "Error: Please make sure the data of input has"+\
            "the same number of embed_dims(in:{0}) with Attention ".format(
            inputs.shape[-1])+"Config(set:{0})  in Attention.".format(
            self.embed_dims)
        
        B, M, D = inputs.shape
        x = self.qkv_proj(inputs) # B, M, 3*num_head*head_dims
        qkv = x.chunk(3, axis=-1) # [[B, M, num_head*head_dims], [...], [...]]
        q, k, v = self.transpose_qkv(qkv) # q/k/v_shape: B, num_head, M, head_dims

        attn = paddle.matmul(q, k, transpose_y=True) # B, num_head, M, M
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v) # B, num_head, M, head_dims
        z = z.transpose(perm=[0, 2, 1, 3]) # B, M, num_head, head_dims
        z = z.reshape(shape=[B, M, self.num_head*self.head_dims])
        z = self.out(z)
        z = self.out_dropout(z)

        return z


class DropPath(nn.Layer):
    """Multi-branch Random Dropout -- Batch randomly dropout different item branches
        Params Info:
            p: Dropout rate
        Forward Tips:
            - Returns the result of a multi-branch random dropout
            - The input data Shape is equal to the output data Shape
    """
    def __init__(self, p=0.):
        super(DropPath, self).__init__(
                 name_scope="droppath")
        assert p >= 0. and p <= 1., \
            "Error: Please make sure the drop_rate is limit at [0., 1.],"+\
            "but now it is {0} in DropPath.".format(p)
        
        self.p = p
    
    def forward(self, inputs):
        if self.p > 0. and self.training:

            keep_p = paddle.to_tensor([1 - self.p], dtype='float32')
            # shape: [B, 1, 1, 1...]
            random_tensor = keep_p + paddle.rand([inputs.shape[0]]+\
                                                 [1]*(inputs.ndim-1),
                                                 dtype='float32')
            random_tensor = random_tensor.floor() # Binarization, round down:[0, 1]
            # Divided by the retention rate
            # it is to ensure that the total expected output during drop remains the same
            output = inputs.divide(keep_p) * random_tensor

            return output

        return inputs


class Former(nn.Layer):
    """Former implementation -- A simple transformer structure
        Params Info:
            embed_dims: Input feature size / embedding dimensions
            mlp_ratio: MLP - Scaling of input features to hidden features
            num_head: The number of Attention head
            qkv_bias: QKV mapping layer whether to use bias, default: True
            dropout_rate: Attentional outcome dropout rate
            droppath_rate: DropPath rate
            attn_dropout_rate: Attentional distribution dropout rate
            mlp_dropout_rate: MLP - Dropout rate
            act: The activation function -- nn.Layer or nn.functional
            norm: Normalized layer -- nn.LayerNorm
        Forward Tips:
            - The input data Shape is equal to the output data Shape
            - Use attention mechanism to calculate global information
    """
    def __init__(self,
                 embed_dims,
                 mlp_ratio=2,
                 num_head=1,
                 qkv_bias=True,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 attn_dropout_rate=0.,
                 mlp_dropout_rate=0.,
                 act=nn.GELU,
                 norm=nn.LayerNorm):
        super(Former, self).__init__(
                 name_scope="former")
        self.embed_dims = embed_dims
        self.mlp_ratio = mlp_ratio
        self.num_head = num_head
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.droppath_rate = droppath_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.mlp_dropout_rate = mlp_dropout_rate

        self.attn = Attention(embed_dims=embed_dims,
                              num_head=num_head,
                              qkv_bias=qkv_bias,
                              dropout_rate=dropout_rate,
                              attn_dropout_rate=attn_dropout_rate)
        self.mlp = MLP(in_features=embed_dims,
                       hidden_features=int(mlp_ratio*embed_dims),
                       dropout_rate=mlp_dropout_rate,
                       act=act)
        
        self.attn_droppath =  DropPath(p=droppath_rate)
        self.mlp_droppath =  DropPath(p=droppath_rate)

        self.attn_norm = norm(embed_dims)
        self.mlp_norm = norm(embed_dims)

    def forward(self, inputs):
        assert inputs.shape[-1] == self.embed_dims, \
            "Error: Please make sure the data of input has"+\
            "the same number of embed_dims(in:{0}) with Former ".format(
            inputs.shape[-1])+"Config(set:{0})  in Former.".format(
            self.embed_dims)

        res = inputs
        x = self.attn(inputs)
        x = self.attn_norm(x)
        x = self.attn_droppath(x)
        x = x + res

        res = x
        x = self.mlp(x)
        x = self.mlp_norm(x)
        x = self.mlp_droppath(x)
        x = x + res

        return x


class ToFormer_Bridge(nn.Layer):
    """ToFormer_Bridge implementation -- Mobile-->Former structure
        Params Info:
            in_channels: Number of input feature map channels
            embed_dims: Input feature size / embedding dimensions
            num_head: The number of Attention head
            q_bias: Q mapping layer whether to use bias, default: True
            dropout_rate: Attentional outcome dropout rate
            droppath_rate: DropPath rate
            attn_dropout_rate: Attentional distribution dropout rate
            norm: Normalized layer -- nn.LayerNorm
        Forward Tips:
            - The input feature map is directly used as
              the sequence data of Key and Value
            - The input Token data map to Query data
            - Input Token sequence data for local information exchange
              and output equal-size Token sequence data
            - The local information extracted from the feature graph is
              integrated into the global information of the Token
    """
    def __init__(self,
                 in_channels,
                 embed_dims,
                 num_head=1,
                 q_bias=True,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 attn_dropout_rate=0.,
                 norm=nn.LayerNorm):
        super(ToFormer_Bridge, self).__init__(
                   name_scope="former_bridge")
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_head = num_head
        self.head_dims = in_channels // num_head
        self.scale = self.head_dims ** -0.5
        self.q_bias = q_bias
        self.dropout_rate = dropout_rate
        self.droppath_rate = droppath_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.q_proj = nn.Linear(in_features=embed_dims,
                                out_features=self.num_head*self.head_dims,
                                bias_attr=q_bias)
        self.out = nn.Linear(in_features=self.num_head*self.head_dims,
                             out_features=embed_dims)

        self.softmax = nn.Softmax()
        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)

        self.out_norm = norm(embed_dims)
        self.out_droppath = DropPath(p=droppath_rate)

    def transpose_q(self, q):
        assert q.ndim == 3, \
            "Error: Please make sure the q has 3 dim,"+\
            " but now it is {0} dim in ToFormer_Bridge-func:transpose_q.".format(
            q.ndim)
        
        B, M, _ = q.shape
        q = q.reshape(shape=[B, M, self.num_head, self.head_dims]).transpose(
                      perm=[0, 2, 1, 3])
        
        return q # B, num_head, M, head_dims
    
    def transform_feature_map(self, feature_map):
        assert feature_map.ndim == 4, \
            "Error: Please make sure the feature_map has 4 dim,"+\
            " but now it is {0} dim in ToFormer_Bridge-func:"+\
            "transform_feature_map.".format(feature_map.ndim)
        
        B, C, H, W = feature_map.shape
        feature_map = feature_map.reshape(shape=[B, C, H*W]
                        ).transpose(perm=[0, 2, 1]) # B, L, C or B, N, C
        feature_map = feature_map.reshape(shape=[B, H*W,
                                          self.num_head, self.head_dims])
        feature_map = feature_map.transpose(perm=[0, 2, 1, 3])

        return feature_map # B, num_head, N, head_dims

    def forward(self, feature_map, z_token):
        assert feature_map.shape[1] == self.in_channels, \
            "Error: Please make sure the data of input has "+\
            "the same number of channels(in:{0}) with ToFormer_Bridge ".format(
            feature_map.shape[1])+"Config(set:{0})  in ToFormer_Bridge.".format(
            self.in_channels)
        assert z_token.shape[-1] == self.embed_dims, \
            "Error: Please make sure the data of input has "+\
            "the same number of embed_dims(in:{0}) with ToFormer_Bridge ".format(
            z_token.shape[-1])+"Config(set:{0})  in ToFormer_Bridge.".format(
            self.embed_dims)

        B, C, H, W = feature_map.shape
        B, M, D = z_token.shape

        q = self.q_proj(z_token) # B, M, num_head*head_dims
        q = self.transpose_q(q=q) # B, num_head, M, head_dims
        
        # B, num_head, N, head_dims
        k = self.transform_feature_map(feature_map=feature_map)
        attn = paddle.matmul(q, k, transpose_y=True) # B, num_head, M, N
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # B, num_head, N, head_dims
        v = self.transform_feature_map(feature_map=feature_map)
        z = paddle.matmul(attn, v) # B, num_head, M, head_dims
        z = z.transpose(perm=[0, 2, 1, 3]).reshape(shape=[B, M,
                                                   self.num_head*self.head_dims])
        z = self.out(z)
        z = self.out_dropout(z)

        # Attractive results standard processing
        z = self.out_norm(z)
        z = self.out_droppath(z)
        z = z + z_token

        return z


class ToMobile_Bridge(nn.Layer):
    """ToMobile_Bridge implementation -- Former-->Mobile structure
        Params Info:
            in_channels: Number of input feature map channels
            embed_dims: Input feature size / embedding dimensions
            num_head: The number of Attention head
            kv_bias: KV mapping layer whether to use bias, default: True
            dropout_rate: Attentional outcome dropout rate
            droppath_rate: DropPath rate
            attn_dropout_rate: Attentional distribution dropout rate
            norm: Normalized layer -- nn.LayerNorm
        Forward Tips:
            - The input feature map is directly used as
              the sequence data of Query
            - The token mapping to Key and Value sequence data
            - Input feature map data for global information interaction
              and output equal size feature map data
            - The global information extracted from the Token
              is fused to the local information in the feature map
    """
    def __init__(self,
                 in_channels,
                 embed_dims,
                 num_head=1,
                 kv_bias=True,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 attn_dropout_rate=0.,
                 norm=nn.LayerNorm):
        super(ToMobile_Bridge, self).__init__(
                   name_scope="mobile_bridge")
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_head = num_head
        self.head_dims = in_channels // num_head
        self.scale = self.head_dims ** -0.5
        self.kv_bias = kv_bias
        self.dropout_rate = dropout_rate
        self.droppath_rate = droppath_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.kv_proj = nn.Linear(in_features=embed_dims,
                                 out_features=2*self.num_head*self.head_dims,
                                 bias_attr=kv_bias)
        
        # to keep the number of channels
        # not necessary
        self.keep_information_linear = Identify() if (self.head_dims * self.num_head) !=\
                                       self.in_channels else \
                                       nn.Linear(in_features=self.num_head*self.head_dims,
                                                 out_features=in_channels)

        self.softmax = nn.Softmax()
        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)

        self.out_norm = norm(in_channels)
        self.out_droppath = DropPath(p=droppath_rate)

    def transpose_kv(self, kv):
        assert len(kv) == 2, \
            "Error: Please make sure the kv has 2 item,"+\
            " but now it is {0} item in ToMobile_Bridge-func:".format(
            len(kv))+"transpose_kv."

        k, v = kv
        B, M, _ = k.shape
        # B, num_head, M, head_dims
        k = k.reshape(shape=[B, M, self.num_head, self.head_dims]).transpose(
                      perm=[0, 2, 1, 3])
        v = v.reshape(shape=[B, M, self.num_head, self.head_dims]).transpose(
                      perm=[0, 2, 1, 3])
        
        return k, v

    def transform_feature_map(self, feature_map):
        assert feature_map.ndim == 4, \
            "Error: Please make sure the feature_map has 4 dim,"+\
            " but now it is {0} dim in ToMobile_Bridge-func:"+\
            "transform_feature_map.".format(feature_map.ndim)

        B, C, H, W = feature_map.shape
        feature_map = feature_map.reshape(shape=[B, C, H*W]
                        ).transpose(perm=[0, 2, 1]) # B, L, C or B, N, C
        feature_map = feature_map.reshape(shape=[B, H*W,
                                          self.num_head, self.head_dims])
        feature_map = feature_map.transpose(perm=[0, 2, 1, 3])

        return feature_map # B, num_head, N, head_dims

    def forward(self, feature_map, z_token):
        assert feature_map.shape[1] == self.in_channels, \
            "Error: Please make sure the data of input has "+\
            "the same number of channels(in:{0}) with ToMobile_Bridge ".format(
            feature_map.shape[1])+"Config(set:{0})  in ToMobile_Bridge.".format(
            self.in_channels)
        assert z_token.shape[-1] == self.embed_dims, \
            "Error: Please make sure the data of input has "+\
            "the same number of embed_dims(in:{0}) with ToMobile_Bridge ".format(
            z_token.shape[-1])+"Config(set:{0})  in ToMobile_Bridge.".format(
            self.embed_dims)

        B, C, H, W = feature_map.shape # N=H*W
        B, M, D = z_token.shape

        x = self.kv_proj(z_token)
        kv = x.chunk(2, axis=-1)
        k, v = self.transpose_kv(kv=kv) # B, num_head, M, head_dims
        # B, num_head, N, head_dims
        q = self.transform_feature_map(feature_map=feature_map)

        attn = paddle.matmul(q, k, transpose_y=True) # B, num_head, N, M
        attn = attn*self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v) # B, num_head, N, head_dims
        z = z.transpose(perm=[0, 2, 1, 3]) # B, N, num_head, head_dims
        z = z.reshape(shape=[B, H*W, self.num_head*self.head_dims])
        z = self.keep_information_linear(z) # keep the number of channels
        z = self.out_dropout(z)

        # Attractive results standard processing
        z = self.out_norm(z)
        z = self.out_droppath(z)
        z = z.transpose(perm=[0, 2, 1]).reshape(shape=[B, C, H, W])
        z = z + feature_map

        return z


class DY_ReLU(nn.Layer):
    """DY_ReLU implementation -- Token As a coefficient input,
                                 Feature map As an activation input
        Params Info:
            in_channels: Number of input feature map channels
            m: The Token length/sequence length
            k: Dynamic parameter base,
               k=2 --> Each complete set of dynamic parameters is
               included[a_1, a_2]and[b_1, b_2],  so on
            use_mlp: Whether the dynamic parameter mapping layer uses MLP,
                     default:False
            mlp_ratio: MLP hidden feature telescopic scale
            coefs: Calculate the coefficient of [a_1, a_2]and[b_1, b_2],
                   Super parameter, default:[1.0, 0.5]
            const: Calculate the constant of [a_1, a_2]and[b_1, b_2],
                   Super parameter, default:[1.0, 0.0]
            dy_reduce: Scaling ratio of hidden maps for dynamic parameter maps
                       based on the number of channels in the input feature graph
        Forward Tips:
            - Input the feature map as the activation input
            - Input token as a coefficient input
            - 1.The result obtained by the token data is mapped by dynamic parameters,
                Participate in [a_1, a_2]and[b_1, b_2] calculations
            - 2.Each Token element (value) generates a set: [a_1(x), a_2(x)]and[b_1(x), b_2(x)]
            - 3.The Token generation results are interacted
                with the input feature map along the channel dimension
                ——Each raw input generates four activation optional values
            - 4.Max selects the largest as the output of dynamic ReLU
    """
    def __init__(self,
                 in_channels,
                 m=3,
                 k=2,
                 use_mlp=False,
                 mlp_ratio=2,
                 coefs=[1.0, 0.5],
                 const=[1.0, 0.0],
                 dy_reduce=6):
        super(DY_ReLU, self).__init__(
                 name_scope="dy_relu")
        self.in_channels = in_channels
        self.m = m
        self.k = k
        self.use_mlp = use_mlp
        self.mlp_ratio = mlp_ratio
        self.coefs = coefs # [0]:a coefficient, [1]:b coefficient
        self.const = const # [0]:a init constant, [1]:b init constant
        self.dy_reduce = dy_reduce

        # k*in_channels——Is to facilitate the dynamic ReLU calculation
        # along the channel dimension
        self.mid_channels = 2*k*in_channels # Multiply by 2 to construct both a and B

        # Construct the coefficients for each a/ b
        # Since k=2, the parameter groups A and B each have two coefficients
        self.lambda_coefs = paddle.to_tensor([self.coefs[0]]*k+\
                                             [self.coefs[1]]*k,
                                             dtype='float32')
        self.init_consts = paddle.to_tensor([self.const[0]]+\
                                            [self.const[1]]*(2*k-1),
                                            dtype='float32')

        # 这里池化Token数据，所以用一维池化
        self.avg_pool = nn.AdaptiveAvgPool1D(output_size=1)
        self.project = nn.Sequential(
            MLP(in_features=m,
                hidden_features=m*mlp_ratio,
                out_features=in_channels//dy_reduce) if use_mlp is True else \
            nn.Linear(in_features=m, out_features=in_channels//dy_reduce),
            nn.ReLU(),
            MLP(in_features=in_channels//dy_reduce,
                hidden_features=(in_channels//dy_reduce)*mlp_ratio,
                out_features=self.mid_channels) if use_mlp is True else \
            nn.Linear(in_features=in_channels//dy_reduce, out_features=self.mid_channels),
            nn.BatchNorm(self.mid_channels)
        )
    
    def forward(self, feature_map, z_token):
        assert feature_map.shape[1] == self.in_channels, \
            "Error: Please make sure the feature_map of input has "+\
            "the same number of channels(in:{0}) with DY_ReLU ".format(
            feature_map.shape[1])+"Config(set:{0})  in DY_ReLU.".format(
            self.in_channels)
        assert z_token.shape[1] == self.m, \
            "Error: Please make sure the z_token of input has "+\
            "the same number of sequence(in:{0}) with DY_ReLU ".format(
            z_token.shape[1])+"Config(set:{0})  in DY_ReLU.".format(
            self.m)

        B, C, H, W = feature_map.shape
        B, M, _ = z_token.shape
        x = self.avg_pool(z_token) # B, M, 1
        x = x.reshape(shape=[B, M]) # B, M
        x = self.project(x) # B, 2*K*C
        x = x.reshape(shape=[B, C, 2*self.k]) # B, C, 2*K

        # Calculation of dynamic coefficients using hyperparameters
        dy_coef = x * self.lambda_coefs + self.init_consts # B, C, 2*K
        # Among them, 2 * K contains a parameter group and b parameter group
        # a is the coefficient, b is the bias.
        # Dynamic ReLU, activation value calculation function: f(x)_k = a_k*x + b_k
        # Dynamic ReLU interactions on channels are calculated here
        x_perm = feature_map.transpose(perm=[2, 3, 0, 1]) # H, W, B, C
        # Add a minimum dimension - alignment: B, C, 2*K, To ensure broadcast computation
        x_perm = x_perm.unsqueeze(-1) # H, W, B, C, 1

        # implementation: f(x)_k = a_k*x + b_k
        output = x_perm * dy_coef[:, :, :self.k] + dy_coef[:, :, self.k:] # H, W, B, C, K
        # Finally, the activation output is executed and
        # the function is judged to be: max(c_f(x)_k)
        # Max is evaluated down the channel dimension
        output = paddle.max(output, axis=-1) # H, W, B, C
        # Restore the input feature graph form
        output = output.transpose(perm=[2, 3, 0, 1]) # H, W, B, C

        return output


class Mobile(nn.Layer):
    """Mobile implementation -- The feature map serves as the feature input,
                                and the Token serves as the parameter selection input(DY_ReLU)
        Params Info:
            in_channels: Number of input feature map channels
            out_channels: Number of output feature map channels
            t: Channel telescopic scale in the Bottle structure
            m: The Token length/sequence length
            kernel_size: Kernel size
            stride: Stride size
            padding: Padding size
            k: Dynamic parameter base,
               k=2 --> Each complete set of dynamic parameters is
               included[a_1, a_2]and[b_1, b_2],  so on
            use_mlp: Whether the dynamic parameter mapping layer uses MLP,
                     default:False
            coefs: Calculate the coefficient of [a_1, a_2]and[b_1, b_2],
                   Super parameter, default:[1.0, 0.5]
            const: Calculate the constant of [a_1, a_2]and[b_1, b_2],
                   Super parameter, default:[1.0, 0.0]
            dy_reduce: Scaling ratio of hidden maps for dynamic parameter maps
                       based on the number of channels in the input feature graph
            groups: 1x1 Convolution grouping number
            mlp_ratio: MLP hides the scaling scale of features
        Forward Tips:
            - Input feature map as feature input
            - Input token as a parameter selection input (DY_RELU)
            - 1.If the stride is 2, an additional downsampling is added
                --use DepthWise_Conv + BacthNorm2D + DY_ReLU
            - 2.Input the feature graph into the Bottle structure，Sequential by PW，DY_ReLU，DW，DY_ReLU，PW
            - 3.Token parameters enter DY_ReLU to provide dynamic parameters
            - 4.Output the feature map after feature extraction
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 t=4,
                 m=3,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 k=2,
                 use_mlp=False,
                 coefs=[1.0, 0.5],
                 const=[1.0, 0.0],
                 dy_reduce=6,
                 mlp_ratio=2,
                 groups=1):
        super(Mobile, self).__init__(
                 name_scope="mobile")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = t
        self.m = m
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.k = k
        self.use_mlp = use_mlp
        self.coefs = coefs
        self.const = const
        self.dy_reduce = dy_reduce
        self.mlp_ratio = mlp_ratio
        self.groups = groups

        if stride == 2:
            self.downsample = nn.Sequential(
                DepthWise_Conv(in_channels=in_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding),
                nn.BatchNorm2D(in_channels)
            )
            self.downsample_act = DY_ReLU(in_channels=in_channels,
                                            m=m, k=k, use_mlp=use_mlp,
                                            coefs=coefs, const=const,
                                            dy_reduce=dy_reduce,
                                            mlp_ratio=mlp_ratio)
        else:
            self.downsample = Identify()
        
        self.in_pointwise_conv = nn.Sequential(
            PointWise_Conv(in_channels=in_channels,
                           out_channels=int(t * in_channels),
                           groups=groups),
            nn.BatchNorm2D(int(t * in_channels))
        )
        self.in_pointwise_conv_act = DY_ReLU(in_channels=int(t * in_channels),
                                                m=m, k=k, use_mlp=use_mlp,
                                                coefs=coefs, const=const,
                                                dy_reduce=dy_reduce,
                                                mlp_ratio=mlp_ratio)

        self.hidden_depthwise_conv = nn.Sequential(
            DepthWise_Conv(in_channels=int(t * in_channels),
                           kernel_size=kernel_size,
                           stride=1,
                           padding=1),
            nn.BatchNorm2D(int(t * in_channels))
        )
        self.hidden_depthwise_conv_act = DY_ReLU(in_channels=int(t * in_channels),
                                                    m=m, k=k, use_mlp=use_mlp,
                                                    coefs=coefs, const=const,
                                                    dy_reduce=dy_reduce,
                                                    mlp_ratio=mlp_ratio)

        self.out_pointwise_conv = nn.Sequential(
            PointWise_Conv(in_channels=int(t * in_channels),
                           out_channels=out_channels,
                           groups=groups),
            nn.BatchNorm2D(out_channels)
        )

    def forward(self, feature_map, z_token):
        assert feature_map.shape[1] == self.in_channels, \
            "Error: Please make sure the feature_map of input has "+\
            "the same number of channels(in:{0}) with Mobile ".format(
            feature_map.shape[1])+"Config(set:{0})  in Mobile.".format(
            self.in_channels)
        assert z_token.shape[1] == self.m, \
            "Error: Please make sure the z_token of input has "+\
            "the same number of sequence(in:{0}) with Mobile ".format(
            z_token.shape[1])+"Config(set:{0})  in Mobile.".format(
            self.m)

        x = self.downsample(feature_map)
        if self.stride == 2:
            x = self.downsample_act(x, z_token)
        x = self.in_pointwise_conv(x)
        x = self.in_pointwise_conv_act(x, z_token)
        x = self.hidden_depthwise_conv(x)
        x = self.hidden_depthwise_conv_act(x, z_token)
        x = self.out_pointwise_conv(x)

        return x


class Stem(nn.Layer):
    """Stem Layer -- Hardswish activation function same as mobilenetV3
        Params Info:
            in_channels: Number of input feature map channels
            out_channels: Number of output feature map channels
            kernel_size: Kernel size
            stride: Stride size
            padding: Padding size
            act: The activation function -- nn.Layer or nn.functional
        Forward Tips:
            - Can replace the activation function directly -- nn.ReLU...
            - Dynamic ReLU requires internal configuration.
              Setting ACT directly does not work
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.Hardswish):
        super(Stem, self).__init__(
                 name_scope="stem")
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

        self.bn = nn.BatchNorm2D(out_channels)
        self.act = act()
    
    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Error: Please make sure the data of input has "+\
            "the same number of channels(in:{0}) with Stem ".format(
            inputs.shape[1])+"Config(set:{0})  in Stem.".format(
            self.in_channels)

        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)

        return x


class Lite_BottleNeck(nn.Layer):
    """Lite BottleNeck -- Conveying decomposition for DepthWise_Conv，use act ReLU6
        Params Info:
            in_channels: Number of input feature map channels
            hidden_channels: Number of hidden feature map channels
            out_channels: Number of output feature map channels
            expands: Scale of the passage in the depth convolution,
                     in_channels + (hidden_channels-in_channels)* 1. / expands
            kernel_size: Kernel size
            stride: Stride size
            padding: Padding size
            act: The activation function -- nn.Layer or nn.functional
            groups: 1x1 Convolution grouping number
        Forward Tips:
            - Can replace the activation function directly -- nn.ReLU...
            - Depthwise_conv is decomposed, so calculates a lighter amount
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 expands=2,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.ReLU6,
                 groups=1):
        super(Lite_BottleNeck, self).__init__(
                      name_scope="lite_bneck")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.expands = expands
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        first_filter = [kernel_size, 1]
        first_stride = [stride, 1]
        first_padding = [padding, 0]
        second_filter = [1, kernel_size]
        second_stride = [1, stride]
        second_padding = [0, padding]

        expands_channels = hidden_channels - in_channels
        self.lite_depthwise_conv = nn.Sequential(
            nn.Conv2D(in_channels=in_channels,
                      out_channels=hidden_channels - expands_channels // expands,
                      kernel_size=first_filter,
                      stride=first_stride,
                      padding=first_padding,
                      groups=self.Gcd(in_channels=in_channels,
                                      hidden_channels=hidden_channels - \
                                          expands_channels // expands)),
            nn.BatchNorm2D(hidden_channels - \
                            expands_channels // expands),
            nn.Conv2D(in_channels=hidden_channels - \
                                  expands_channels // expands,
                      out_channels=hidden_channels,
                      kernel_size=second_filter,
                      stride=second_stride,
                      padding=second_padding,
                      groups=self.Gcd(hidden_channels=hidden_channels,
                                      in_channels=hidden_channels - \
                                          expands_channels // expands)),
            nn.BatchNorm2D(hidden_channels),
            act()
        )

        self.pointwise_conv = nn.Sequential(
            PointWise_Conv(in_channels=hidden_channels,
                           out_channels=out_channels,
                           groups=groups),
            nn.BatchNorm2D(out_channels),
        )

    def Gcd(self, in_channels, hidden_channels):
        """European miles to calculate the maximum number of groups (Causes)
        """
        m = in_channels
        n = hidden_channels
        r = 0
        while n > 0:
            r = m % n
            m = n
            n = r

        return m
    
    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Error: Please make sure the data of input has"+\
            "the same number of channels(in:{0}) with Lite_BottleNeck ".format(
            inputs.shape[1])+"Config(set:{0})  in Lite_BottleNeck.".format(
            self.in_channels)
        
        x = self.lite_depthwise_conv(inputs)
        x = self.pointwise_conv(x)

        return x


class Classifier_Head(nn.Layer):
    """Classifier_Head -- Output classification results
        Params Info:
            in_channels: Number of input feature map channels
            embed_dims: Output token embedding dimensions
            expands: Lite_BottleNeck - Scale of the passage in the depth convolution
            num_classes: Number of class
            head_type: Classification header type,
                       Automatically adjust the hidden layer size of different models
            act: The activation function -- nn.Layer or nn.functional
        Forward Tips:
            - Can replace the activation function directly -- nn.ReLU...
    """
    def __init__(self,
                 in_channels,
                 embed_dims,
                 num_classes,
                 head_type='base',
                 act=nn.Hardswish):
        super(Classifier_Head, self).__init__(
                            name_scope="head")
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.head_type = head_type

        self.head_hidden_size = get_head_hidden_size(head_type)

        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc = nn.Linear(in_features=in_channels + embed_dims,
                            out_features=self.head_hidden_size)
        self.out = nn.Linear(in_features=self.head_hidden_size,
                            out_features=num_classes)

        self.flatten = nn.Flatten()
        self.act = act()
        self.softmax = nn.Softmax()
    
    def forward(self, feature_map, z_token):
        assert feature_map.shape[1] == self.in_channels, \
            "Error: Please make sure the feature_map of input has "+\
            "the same number of channels(in:{0}) with Classifier_Head ".format(
            feature_map.shape[1])+"Config(set:{0})  in Classifier_Head.".format(
            self.in_channels)
        assert z_token.shape[-1] == self.embed_dims, \
            "Error: Please make sure the z_token of input has "+\
            "the same number of embed_dims(in:{0}) with Classifier_Head ".format(
            z_token.shape[-1])+"Config(set:{0})  in Classifier_Head.".format(
            self.embed_dims)

        x = self.avg_pool(feature_map)
        x = self.flatten(x) # B, C

        cls_token = z_token[:, 0] # B, D
        x = paddle.concat([x, cls_token], axis=-1) # B, C+D

        x = self.fc(x)
        x = self.act(x)
        x = self.out(x)
        x = self.softmax(x)

        return x


class Basic_Block(nn.Layer):
    """MobileFormer Minimum base module
        Params Info:
            embed_dims: Token embedding dimensions
            in_channels: Number of input feature map channels
            out_channels: Number of output feature map channels
            num_head: Number of Attention head,
                      default:1
            mlp_ratio: MLP hidden feature telescopic scale,
                       default:2
            q_bias: Mobile-->Former whether use bias,
                    default:True
            kv_bias: Former-->Mobile whether use bias,
                     default:True
            qkv_bias: former whether use bias,
                      default:True
            dropout_rate: Attentional outcome dropout rate,
                          default:0.
            droppath_rate: DropPath rate, default:0.
            attn_dropout_rate: Attentional dropout rate, default:0.
            mlp_dropout_rate: Multi-level perception machine discard rate,
                              default:0.
            t: Hidden channel telescopic ratio in
               deep separable convolution, default:1
            m: Token sequence length, default:6
            kernel_size: Kernel size, default:3
            stride: Stride size, default:1
            padding: Padding size, default:0
            k: Number of dynamic ReLU parameters, default:2
            use_mlp: Whether to use mlp to map
                     when calculating dynamic parameters of dynamic RELU,
                     default:False
            coefs: Dynamic parameter coefficient, default:[1.0, 0.5]
            const: Dynamic parameters initial constant, default:[1.0, 0.0]
            dy_reduce: Dynamic parameter mapping hides feature scaling scale,
                       default:6
            groups: 1x1 Convolution grouping number, default:1
            add_pointwise_conv: Whether or not to add an additional 1x1 convolution,
                                default:False
            pointwise_conv_channels: Number of additional convolutional
                                     expansion channels, default:None,
                                     If add_pointwise_conv is not false,
                                     the value should be filled.
            act: The activation function -- nn.Layer or nn.functional
            norm: Normalized layer -- nn.LayerNorm
        Forward Tips:
            - ToFormer_Bridge --> Former --> Mobile --> ToMobile_Bridge
    """
    def __init__(self,
                 embed_dims,
                 in_channels,
                 out_channels,
                 num_head=1,
                 mlp_ratio=2,
                 q_bias=True,
                 kv_bias=True,
                 qkv_bias=True,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 attn_dropout_rate=0.,
                 mlp_dropout_rate=0.,
                 t=1,
                 m=6,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 k=2,
                 use_mlp=False,
                 coefs=[1.0, 0.5],
                 const=[1.0, 0.0],
                 dy_reduce=6,
                 groups=1,
                 add_pointwise_conv=False,
                 pointwise_conv_channels=None,
                 act=nn.GELU,
                 norm=nn.LayerNorm):
        super(Basic_Block, self).__init__(
                 name_scope="basic_block")
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_head = num_head
        self.mlp_ratio = mlp_ratio
        self.q_bias = q_bias
        self.kv_bias = kv_bias
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.droppath_rate = droppath_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.mlp_dropout_rate = mlp_dropout_rate
        self.t = t
        self.m = m
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.k = k
        self.use_mlp = use_mlp
        self.coefs = coefs
        self.const = const
        self.dy_reduce = dy_reduce
        self.use_mlp = use_mlp
        self.add_pointwise_conv = add_pointwise_conv
        self.pointwise_conv_channels = pointwise_conv_channels
        self.groups = groups

        self.mobile = Mobile(in_channels=in_channels,
                             out_channels=out_channels,
                             t=t, m=m, kernel_size=kernel_size,
                             stride=stride, padding=padding, k=k,
                             use_mlp=use_mlp, coefs=coefs,
                             const=const, dy_reduce=dy_reduce,
                             mlp_ratio=mlp_ratio, groups=groups)

        self.toformer_bridge = ToFormer_Bridge(in_channels=in_channels,
                                               embed_dims=embed_dims,
                                               num_head=num_head,
                                               q_bias=q_bias,
                                               dropout_rate=dropout_rate,
                                               droppath_rate=droppath_rate,
                                               attn_dropout_rate=attn_dropout_rate,
                                               norm=norm)
        
        self.former = Former(embed_dims=embed_dims,
                             mlp_ratio=mlp_ratio,
                             num_head=num_head,
                             qkv_bias=qkv_bias,
                             dropout_rate=dropout_rate,
                             droppath_rate=droppath_rate,
                             attn_dropout_rate=attn_dropout_rate,
                             mlp_dropout_rate=mlp_dropout_rate,
                             act=act,
                             norm=norm)
        
        self.tomobile_bridge = ToMobile_Bridge(in_channels=out_channels,
                                               embed_dims=embed_dims,
                                               num_head=num_head,
                                               kv_bias=kv_bias,
                                               dropout_rate=dropout_rate,
                                               droppath_rate=droppath_rate,
                                               attn_dropout_rate=attn_dropout_rate,
                                               norm=norm)
        
        if add_pointwise_conv == True:
            self.extra_toformer_bridge = ToFormer_Bridge(in_channels=out_channels,
                                                        embed_dims=embed_dims,
                                                        num_head=num_head,
                                                        q_bias=q_bias,
                                                        dropout_rate=dropout_rate,
                                                        droppath_rate=droppath_rate,
                                                        attn_dropout_rate=attn_dropout_rate,
                                                        norm=norm)
            self.pointwise_conv = PointWise_Conv(in_channels=out_channels,
                                                 out_channels=pointwise_conv_channels,
                                                 groups=groups)
        else:
            self.pointwise_conv = Identify()
    
    def forward(self, feature_map, z_token):
        assert feature_map.shape[1] == self.in_channels, \
            "Error: Please make sure the feature_map of input has "+\
            "the same number of channels(in:{0}) with Basic_Block ".format(
            feature_map.shape[1])+"Config(set:{0})  in Basic_Block.".format(
            self.in_channels)
        assert z_token.shape[-1] == self.embed_dims, \
            "Error: Please make sure the z_token of input has "+\
            "the same number of embed_dims(in:{0}) with Basic_Block ".format(
            z_token.shape[-1])+"Config(set:{0})  in Basic_Block.".format(
            self.embed_dims)

        z = self.toformer_bridge(feature_map, z_token)
        z = self.former(z)

        x = self.mobile(feature_map, z)
        output_x = self.tomobile_bridge(x, z)

        if self.add_pointwise_conv == True:
            # The last global interaction incorporating local information
            z = self.extra_toformer_bridge(output_x, z)

        output_x = self.pointwise_conv(output_x)

        return output_x, z


class MobileFormer(nn.Layer):
    """MobileFormer Network implementation
    """
    def __init__(self,
                 num_classes=1000,
                 in_channels=3,
                 model_type='base',
                 alpha=1.0,
                 print_info=False):
        super(MobileFormer, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.model_type = model_type
        self.alpha = alpha

        self.net_configs = MobileFormer_Config(model_type=model_type,
                                               alpha=alpha, print_info=print_info)
        self.build_model() # Building a basic configuration

        self.global_token = self.create_parameter(shape=[1, self.m, self.embed_dims],
                                dtype='float32', attr=nn.initializer.KaimingUniform())

        self.stem = self.create_stem()

        self.bneck_lite = self.create_bneck_lite()

        self.mf_blocks = self.create_mf_blocks()

        self.head = self.create_head()

    def build_model(self):
        """Building a basic configuration
        """
        base_cfg = self.net_configs['base_cfg']
        self.m = base_cfg[0]
        self.embed_dims = base_cfg[1]
        self.num_head = base_cfg[2]
        self.mlp_ratio = base_cfg[3]
        self.q_bias = base_cfg[4]
        self.kv_bias = base_cfg[5]
        self.qkv_bias = base_cfg[6]
        self.dropout_rate = base_cfg[7]
        self.droppath_rate = base_cfg[8]
        self.attn_dropout_rate = base_cfg[9]
        self.mlp_dropout_rate = base_cfg[10]
        self.k = base_cfg[11]
        self.use_mlp = base_cfg[12]
        self.coefs = base_cfg[13]
        self.const = base_cfg[14]
        self.dy_reduce = base_cfg[15]
        self.groups = base_cfg[16]

    def create_stem(self):
        """Construct a gradual entry
        """
        return Stem(in_channels=self.net_configs['stem'][0],
                    out_channels=self.net_configs['stem'][1],
                    kernel_size=self.net_configs['stem'][2],
                    stride=self.net_configs['stem'][3],
                    padding=self.net_configs['stem'][4],
                    act=self.net_configs['stem'][5])

    def create_bneck_lite(self):
        """Build a Bottleneck-Lite structure
        """
        return Lite_BottleNeck(in_channels=self.net_configs['bneck-lite'][0],
                               hidden_channels=self.net_configs['bneck-lite'][1],
                               out_channels=self.net_configs['bneck-lite'][2],
                               expands=self.net_configs['bneck-lite'][3],
                               kernel_size=self.net_configs['bneck-lite'][4],
                               stride=self.net_configs['bneck-lite'][5],
                               padding=self.net_configs['bneck-lite'][6],
                               act=self.net_configs['bneck-lite'][7],
                               groups=self.groups)

    def create_mf_blocks(self):
        """Construction of MF(basic) block
        """
        blocks = []
        blocks_config = self.net_configs['MF']
        blocks_depth = len(blocks_config)

        for i in range(blocks_depth):
            blocks.append(
                Basic_Block(embed_dims=self.embed_dims,
                            in_channels=blocks_config[i][0],
                            out_channels=blocks_config[i][1],
                            num_head=self.num_head,
                            mlp_ratio=self.mlp_ratio,
                            q_bias=self.q_bias,
                            kv_bias=self.kv_bias,
                            qkv_bias=self.qkv_bias,
                            dropout_rate=self.dropout_rate,
                            droppath_rate=self.droppath_rate,
                            attn_dropout_rate=self.attn_dropout_rate,
                            mlp_dropout_rate=self.mlp_dropout_rate,
                            t=blocks_config[i][2],
                            m=self.m,
                            kernel_size=blocks_config[i][3],
                            stride=blocks_config[i][4],
                            padding=blocks_config[i][5],
                            k=self.k,
                            use_mlp=self.use_mlp,
                            coefs=self.coefs,
                            const=self.const,
                            dy_reduce=self.dy_reduce,
                            add_pointwise_conv=blocks_config[i][6],
                            pointwise_conv_channels=blocks_config[i][7],
                            act=blocks_config[i][8],
                            norm=blocks_config[i][9],
                            groups=self.groups)
            )

        blocks = nn.LayerList(blocks)

        return blocks

    def create_head(self):
        """Create category header
        """
        return Classifier_Head(in_channels=self.net_configs['head'][0],
                               embed_dims=self.embed_dims,
                               num_classes=self.num_classes,
                               head_type=self.net_configs['head'][1],
                               act=self.net_configs['head'][2]
                              )

    def create_token(self, B):
        """Build the appropriate Token sequence
           tensor for the Batch data
        """
        z_token = paddle.concat([self.global_token]*B)
        return z_token

    def forward(self, inputs):
        z_token = self.create_token(inputs.shape[0])

        x = self.stem(inputs)
        x = self.bneck_lite(x)

        for b in self.mf_blocks:
            x, z_token = b(x, z_token)
        
        x = self.head(x, z_token)

        return x


def MobileFormer_Big(num_classes,
                      in_channels,
                      alpha=1.0, print_info=False):
    """Build a BIG-MobileForMer model
    """
    return MobileFormer(num_classes=num_classes,
                        in_channels=in_channels,
                        model_type='big',
                        alpha=alpha, print_info=print_info)

def MobileFormer_Base(num_classes,
                      in_channels,
                      alpha=1.0, print_info=False):
    """Build a base-mobileformer model
    """
    return MobileFormer(num_classes=num_classes,
                        in_channels=in_channels,
                        model_type='base',
                        alpha=alpha, print_info=print_info)

def MobileFormer_Base_Small(num_classes,
                      in_channels,
                      alpha=1.0, print_info=False):
    """Build a Base_Small-MobileForMER model
    """
    return MobileFormer(num_classes=num_classes,
                        in_channels=in_channels,
                        model_type='base_small',
                        alpha=alpha, print_info=print_info)

def MobileFormer_Mid(num_classes,
                      in_channels,
                      alpha=1.0, print_info=False):
    """Build a MID-MobileForMer model
    """
    return MobileFormer(num_classes=num_classes,
                        in_channels=in_channels,
                        model_type='mid',
                        alpha=alpha, print_info=print_info)

def MobileFormer_Mid_Small(num_classes,
                      in_channels,
                      alpha=1.0, print_info=False):
    """Build a MID_SMALL-MobileForMER model
    """
    return MobileFormer(num_classes=num_classes,
                        in_channels=in_channels,
                        model_type='mid_small',
                        alpha=alpha, print_info=print_info)

def MobileFormer_Small(num_classes,
                      in_channels,
                      alpha=1.0, print_info=False):
    """Build a Small-MobileForMer model
    """
    return MobileFormer(num_classes=num_classes,
                        in_channels=in_channels,
                        model_type='small',
                        alpha=alpha, print_info=print_info)

def MobileFormer_Tiny(num_classes,
                      in_channels,
                      alpha=1.0, print_info=False):
    """Build a Tiny-MobileForMer model
    """
    return MobileFormer(num_classes=num_classes,
                        in_channels=in_channels,
                        model_type='tiny',
                        alpha=alpha, print_info=print_info)


if __name__ == "__main__":
    data = paddle.empty((1, 3, 224, 224))
    model = MobileFormer_Base(num_classes=1000, in_channels=3)
    # y_pred = model(data)
    # print(y_pred.shape)

    # summary
    # model = paddle.Model(model)
    # print(model.summary(input_size=(1, 3, 224, 224)))

    # flops
    print(paddle.flops(model, input_size=(1, 3, 224, 224),
                       print_detail=True))
