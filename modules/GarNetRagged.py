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
            B : batch size
            H : number of hits
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
            self.input_feature_transform = tf.keras.layers.Dense(n_FLR_nodes, name="FLR")

        with tf.name_scope(self.name+"S"):
            self.aggregator_distance = tf.keras.layers.Dense(n_aggregators, name="S")
        
        with tf.name_scope(self.name+"F_out"):
            self.output_feature_transform = tf.keras.layers.Dense(n_Fout_nodes, activation="tanh", name="F_out")

    def build(self, input_shape):
        input_shape = input_shape[0]

        with tf.name_scope(self.name+"F_LR"):
            self.input_feature_transform.build(input_shape)

        with tf.name_scope(self.name+"S"):
            self.aggregator_distance.build(input_shape)

        with tf.name_scope(self.name+"F_out"):
            # F_out (B*H, F+2*S*P)
            self.output_feature_transform.build((
                input_shape[0],
                input_shape[1] + 2*self.n_FLR_nodes*self.n_aggregators))

        super().build(input_shape)

    def call(self, inputs):
        x, rs = inputs
        # d distance between H vertices and S aggregators
        # d = Dense(B*H, F) = (B*H, S)
        distance = self.aggregator_distance(x)
        # Matrix V(d_jk: (B*H, S)
        edge_weights = tf.exp(-distance**2)
        # F_LR: rs(Dense(B*H, F)) = (B*H, P)
        # Has same shape as x but learned representation therefore new values
        features_LR = self.input_feature_transform(x)
        # f_tilde: f_ij x V(d_jk) = (B*H, 1, P) x (B*H, S, 1) = (B*H, S, P)
        f_tilde = tf.expand_dims(features_LR, axis=1) * tf.expand_dims(edge_weights, axis=2)
        # f_tilde: rs(Dense(B*H, S, P)) = (B, H, S, P)
        f_tilde = tf.RaggedTensor.from_row_splits(f_tilde, rs)
        # Aggregation of f_tilde (B, S, 2*P)
        f_tilde_aggregated = tf.concat([
            tf.math.reduce_mean(f_tilde+1E-10, axis=1),
            tf.math.reduce_max(f_tilde, axis=1)],
            axis=2)
        # Return f_updated to the hits: (B, S, 2*P) x (B*H, S, 1)  = (B*H, S, 2*P)
        f_updated = f_tilde_aggregated * tf.expand_dims(edge_weights, axis=2)
        # Reshape to (B*H, 2*P*S)
        f_updated = tf.reshape(
            f_updated,
            [tf.shape(f_updated)[0], 2*self.n_FLR_nodes*self.n_aggregators])
        # Feature vector as (B*H, F+2*P*S)
        f_out = tf.concat(
            [x, f_updated],
            axis=1)
        f_out = self.output_feature_transform(f_out)

        return f_out, distance

    def get_config(self):
        config = {
            "n_aggregators": self.n_aggregators,
            "n_Fout_nodes": self.n_Fout_nodes,
            "n_FLR_nodes": self.n_FLR_nodes,
            "name": self.name}
        return dict(list(super().get_config().items()) + list(config.items()))
