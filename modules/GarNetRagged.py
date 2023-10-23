import tensorflow as tf
from baseModules import LayerWithMetrics


class GarNetRagged(LayerWithMetrics):
    """ GarNet layer for Ragged Tensors

    Abbreviations for shapes:
        B : batch size
        H : number of hits
        F : number of features
        S : number of aggregators
    """
    def __init__(
            self,
            n_aggregators,
            n_filters,
            n_propagate,
            name,
            **kwargs):
        super().__init__(**kwargs)

        self.input_feature_transform = tf.keras.layers.Dense(n_propagate, name=f"{name}_FLR")
        self.aggregator_distance = tf.keras.layers.Dense(n_aggregators, name=f"{name}_S")
        self.output_feature_transform = tf.keras.layers.Dense(n_filters, activation="tanh", name=f"{name}_Fout")

        self._sublayers = [
            self.input_feature_transform,
            self.aggregator_distance,
            self.output_feature_transform
        ]

    def build(self, input_shape):
        self.input_feature_transform.build(input_shape)
        self.aggregator_distance.build(input_shape)
        self.output_feature_transform.build(input_shape)  # TODO Is not correct yet

        for layer in self._sublayers:
            self._trainable_weights += layer.trainable_weights
            self._non_trainable_weights += layer.non_trainable_weights

        super().build(input_shape)

    def call(self, inputs):
        x = tf.RaggedTensor.from_row_splits(inputs[0], inputs[1])  # (B, H, F)
        edge_weights = tf.exp(-self.aggregator_distance(x)**2)  # (B, H, S)
        features_with_edge_weights = tf.concat([
            self.input_feature_transform(x),
            edge_weights
        ], axis=-1)  # (B, H, F+S)
        edge_weights_transposed = tf.transpose(edge_weights, perm=[0, 2, 1])  # (B, S, H)

        aggregated = tf.concat([
            self.apply_edge_weights(
                features_with_edge_weights,
                edge_weights_transposed,
                aggregation=tf.reduce_max),  # (B, S, H, 1) x (B, 1, H, F+S) = (B, S, H, F+S), max(axis=2) -> (B, S, F+S)
            self.apply_edge_weights(
                features_with_edge_weights,
                edge_weights_transposed,
                aggregation=tf.reduce_mean),  # (B, S, H, 1) x (B, 1, H, F+S) = (B, S, H, F+S), mean(axis=2) -> (B, S, F+S)
        ], axis=-1)  # (B, S, 2*(F+S))

        features_updated = tf.concat([
            x,
            self.apply_edge_weights(aggregated, edge_weights),
            edge_weights
        ], axis=-1)  # (B, H, F) + (B, 1, S, 2*(F+S)) x (B, H, S, 1) + (B, H, S) = (B, H, F) + (B, H, S, 2*(F+S)) + (B, H, S)

        return self.output_feature_transform(features_updated)

    def apply_edge_weights(
            self,
            features,
            edge_weights,
            aggregation=None):
        features = tf.expand_dims(features, axis=1)
        edge_weights = tf.expand_dims(edge_weights, axis=3)

        out = edge_weights*features
        n = features.shape[2].value*features.shape[3].value

        if aggregation:
            out = aggregation(out, axis=2)
            n = features

        return tf.reshape(out, )  # TODO
          
    def get_config(self):
        config = {
            "n_aggregators": self.n_aggregators,
            "n_filters": self.n_filters,
            "n_propagate": self.n_propagate,
            "name": self.name
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))



    