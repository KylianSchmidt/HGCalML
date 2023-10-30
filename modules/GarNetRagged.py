import tensorflow as tf
from baseModules import LayerWithMetrics


class GarNetRagged(LayerWithMetrics):
    def __init__(
            self,
            n_aggregators: int,
            n_Fout_nodes: int,
            n_FLR_nodes: int,
            **kwargs):
        """ GarNet layer for Ragged Tensors, following the structure detailled in
        https://arxiv.org/pdf/1902.07987.pdf

        Abbreviations for shapes:
            E : number of events
            H : number of hits
            B : batch size = E*H
            F : number of features
            S : number of aggregators
            P : number of propagators (n_FLR_nodes)
            rs: row splits
        """
        super().__init__(**kwargs)
        self.n_aggregators = n_aggregators
        self.n_Fout_nodes = n_Fout_nodes
        self.n_FLR_nodes = n_FLR_nodes

        with tf.name_scope(self.name+"F_LR"):
            self.input_feature_transform = tf.keras.layers.Dense(
                n_FLR_nodes, name="FLR")

        with tf.name_scope(self.name+"S"):
            self.aggregator_distance = tf.keras.layers.Dense(
                n_aggregators, name="S")

        with tf.name_scope(self.name+"F_out"):
            self.output_feature_transform = tf.keras.layers.Dense(
                n_Fout_nodes, activation="tanh", name="F_out")

    def build(self, input_shape):
        input_shape = input_shape[0]

        with tf.name_scope(self.name+"F_LR"):
            self.input_feature_transform.build(input_shape)

        with tf.name_scope(self.name+"S"):
            self.aggregator_distance.build(input_shape)

        with tf.name_scope(self.name+"F_out"):
            # F_out (E*H, F+2*S*P)
            self.output_feature_transform.build((
                input_shape[0],
                input_shape[1] + 1*self.n_FLR_nodes*self.n_aggregators))  # TODO change to 2

        super().build(input_shape)

    def call(self, inputs):
        x, rs = inputs
        # d distance between H vertices and S aggregators
        # d: Dense(E*H, F) = (E*H, S)
        distance = self.aggregator_distance(x)
        # Matrix V(d_jk: (E*H, S)
        edge_weights = tf.exp(-distance**2)
        # F_LR: rs(Dense(E*H, F)) = (E*H, P)
        # Has same shape as x but learned representation therefore new values
        features_LR = self.input_feature_transform(x)
        # f_tilde: f_ij x V(d_jk) = (E*H, 1, P) x (E*H, S, 1) = (E*H, S, P)
        f_tilde = tf.expand_dims(features_LR, axis=1) * tf.expand_dims(edge_weights, axis=2)
        # f_tilde: rs(Dense(E*H, S, P)) = (E, H, S, P)
        f_tilde = tf.RaggedTensor.from_row_splits(f_tilde, rs)
        # f_tilde_mean (E, 1, S, P)
        f_tilde_mean = tf.reduce_mean(f_tilde, axis=1, keepdims=False)

        # rs(E*H, S) = (E, H, S)
        edge_weights = tf.RaggedTensor.from_row_splits(edge_weights, rs)

        rl = edge_weights.row_lengths()

        # Repeat
        f_tilde_mean = tf.repeat(f_tilde_mean, rl, axis=0)

        # f_tilde_mean (E, 1, S, P)
        f_tilde_mean = tf.RaggedTensor.from_row_lengths(
            values=f_tilde_mean,
            row_lengths=rl
            ).with_row_splits_dtype(tf.int32)
        print("F_tilde", f_tilde_mean.shape)

        # edge_weights (E, H, S, 1)
        edge_weights = tf.expand_dims(edge_weights, axis=3)
        print("EDGE_WEIGHTS SHAPE", edge_weights.shape)

        # Return f_updated to the hits: (E, 1, S, P) x (E, H, S, 1) = (E, H, S, P)
        f_updated = f_tilde_mean * edge_weights

        # Reshape to (E, H, P*S)
        f_updated = f_updated.merge_dims(2, 3)
        f_updated = f_updated.merge_dims(0, 1)
        # Feature vector as (E*H, F+P*S)
        f_out = tf.concat(
            [x, f_updated],
            axis=1)
        f_out = self.output_feature_transform(f_out)

        return f_out, distance

    def take_mean(self, input: tf.RaggedTensor, axis: int):
        # (E, H, S, P) -> (E, H, S, P)
        return 0

    def get_config(self):
        config = {
            "n_aggregators": self.n_aggregators,
            "n_Fout_nodes": self.n_Fout_nodes,
            "n_FLR_nodes": self.n_FLR_nodes,
            "name": self.name
        }
        return dict(list(super().get_config().items()) + list(config.items()))
