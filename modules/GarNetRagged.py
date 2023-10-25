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
            P : number of propagators
            rs: row splits
        """
        super().__init__(**kwargs)
        self.n_aggregators = n_aggregators
        self.n_Fout_nodes = n_Fout_nodes
        self.n_FLR_nodes = n_FLR_nodes

        self.input_feature_transform = tf.keras.layers.Dense(n_FLR_nodes, name="FLR")
        self.aggregator_distance = tf.keras.layers.Dense(n_aggregators, name="S")
        self.output_feature_transform = tf.keras.layers.Dense(n_Fout_nodes, activation="tanh", name="Fout")

        self._sublayers = [
            self.input_feature_transform,
            self.aggregator_distance,
            self.output_feature_transform
        ]

    def build(self, input_shape):
        print("Input shape in GarNetRagged.build", input_shape)
        input_shape = input_shape[0]
        self.input_feature_transform.build(input_shape)
        self.aggregator_distance.build(input_shape)
        self.output_feature_transform.build((
            input_shape[0],
            input_shape[1],
            2*self.n_FLR_nodes*self.n_aggregators))

        for layer in self._sublayers:
            self._trainable_weights += layer.trainable_weights
            self._non_trainable_weights += layer.non_trainable_weights

        super().build(input_shape)

    def call(self, inputs):
        x, rs = inputs
        # d distance between H vertices and S aggregators
        # d = Dense(B*H, F) = (B*H, S)
        distance = self.aggregator_distance(x)
        # Matrix V(d_jk: (B, H, S)
        edge_weights = tf.RaggedTensor.from_row_splits(tf.exp(-distance**2), rs)
        # F_LR: rs(Dense(B*H, F)) = rs(B*H, P) = (B, H, P)
        # Has same shape as x but learned representation therefore new values
        features_LR = tf.RaggedTensor.from_row_splits(self.input_feature_transform(x), rs)
        # f_tilde: f_ij x V(d_jk) = (B, H, 1, P) x (B, H, S, 1) = (B, H, S, P)
        f_tilde = features_LR * edge_weights
    #    f_tilde = tf.expand_dims(features_LR, axis=2) * tf.expand_dims(edge_weights, axis=3)
        # Aggregation of f_tilde (B, S, 2*P))
        f_tilde_aggregated = tf.concat([
            tf.reduce_mean(f_tilde+1E-10, axis=1),
            tf.reduce_max(f_tilde, axis=1)],
            axis=-1)
        # Return f_updated to the hits: (B, 1, S, 2*P) x (B, H, S, 1)  = (B, H, S, 2*P)
        print("TESZ", f_tilde.row_splits)
        print("edge_weights", edge_weights.row_splits)
        f_updated = tf.expand_dims(f_tilde_aggregated, axis=1) * tf.expand_dims(edge_weights, axis=3)
        # Reshape to (B, H, 2*P*S)
        f_updated = f_updated.merge_dims(2, 3)
        # Feature vector as (B, H, F+2*P*S)
        f_out = tf.concat([
            tf.RaggedTensor.from_row_splits(x, rs),
            f_updated],
            axis=-1)
        f_out = self.output_feature_transform(f_updated)

        return f_out, distance

    def get_config(self):
        config = {
            "n_aggregators": self.n_aggregators,
            "n_Fout_nodes": self.n_Fout_nodes,
            "n_FLR_nodes": self.n_FLR_nodes,
            "name": self.name}
        return dict(list(super().get_config().items()) + list(config.items()))
