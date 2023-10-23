
import tensorflow as tf
from baseModules import LayerWithMetrics
from Losses import nntr_L2_distance
from Layers import RaggedGlobalExchange
from RaggedLayers import CollapseRagged
from GravNetLayersRagged import (CastRowSplits, ScaledGooeyBatchNorm2, RaggedGravNet)
from callbacks import NanSweeper
from DeepJetCore.training.training_base import training_base
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback


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
        # Learned representation of the vertex features
        self.learned_representation = tf.keras.layers.Dense(
            n_propagate,
            name=f"{name}_FLR")
        # Distance between the considered vertex and the aggregators S
        self.aggregator_distance = tf.keras.layers.Dense(
            n_aggregators,
            name=f"{name}_S")
        # Layer for F_out
        self.output_feature_transform = tf.keras.layers.Dense(
            n_filters,
            activation="tanh",
            name=f"{name}_Fout")

        super().__init__(**kwargs)

    def build(self, input_shape):
        # TODO build the layers

        super().build(input_shape)

    def call(self, inputs):
        # F_in (B, H, F)
        F_in = tf.RaggedTensor.from_row_splits(inputs[0], inputs[1])
        # S, represented by the distance d between vertex and aqgregators (B, H, S)
        distance = self.aggregator_distance(F_in)
        # Matrix V(d_jk) (B, H, S)
        edge_weights = tf.exp(-distance**2)
        # f_ij (B, H, F)
        features_LR = self.learned_representation(F_in)
        # f_tilde: f_ij x V(d_jk) = (B, H, 1, F) x (B, H, S, 1) = (B, H, S, F)
        f_tilde = (tf.expand_dims(features_LR, axis=2)
                   * tf.expand_dims(edge_weights, axis=3))
        # f_tilde_average and max (B, S, F)
        f_tilde_average = tf.reduce_mean(f_tilde, axis=1)
        f_tilde_max = tf.reduce_max(f_tilde, axis=1)
        # Aggregation of f_tilde (B, S, 2*F))
        f_tilde_mean_max = tf.concat([f_tilde_average, f_tilde_max], axis=-1)
        # Return f_tilde_aggregated to the hits: (B, 1, S, 2*F) x (B, H, S, 1)  = (B, H, S, 2*F)
        f_updated = (tf.expand_dims(f_tilde_mean_max, axis=1)
                     * tf.expand_dims(edge_weights, axis=2))
        # Reshape to (B, H, 2*F*S)
        f_updated = tf.reshape(f_updated, shape=[-1, f_updated.shape[1], f_updated.shape[2]*f_updated.shape[3]])
        # F_out (B, H, F_out)
        F_out = self.output_feature_transform(
            tf.concat([
                F_in, f_updated, edge_weights
            ], axis=-1))
        return F_out

