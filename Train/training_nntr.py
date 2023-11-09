'''
Track Reconstruction algorithm for the simulation of the LUXE experiment
Uses the GravNet architecture
'''

import os
import tensorflow as tf
import logging
from Losses import nntr_L2_distance
from Layers import RaggedGlobalExchange
from RaggedLayers import CollapseRagged
from GravNetLayersRagged import (CastRowSplits, ScaledGooeyBatchNorm2, RaggedGravNet)
from GarNetRagged import GarNetRagged
from callbacks import NanSweeper
from training_base import TrainingBase
from predict import Prediction
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback

logger = logging.getLogger(__name__)


class NNTR():
    def __init__(
            self,
            detector_type="idealized_detector",
            model_name="v1_test",
            train_uncertainties=False,
            takeweights="",
            ):
        self.detector_type = detector_type
        self.model_name = model_name
        self.train_uncertainties = train_uncertainties
        self.takeweights = takeweights
        self.output_dir = f"./nntr_models/{self.detector_type}/{self.model_name}/Output"
        self.predicted_dir = f"./nntr_models/{self.detector_type}/{self.model_name}/Predicted"

        if os.path.isdir(self.output_dir):
            logger.warning("Output directory already exists, rename 'output_dir' to avoid overwriting")
        if os.path.isdir(self.predicted_dir):
            logger.warning("Predicted directory already exists, rename 'predicted_dir' to avoid overwriting")

    def two_vertex_fitter(inputs):
        """ Network model for the BeamDumpTrackCalo two photon reconstruction

        Notes
        -----
        Specific Model Name:
        GravNet from the original paper https://arxiv.org/pdf/1902.07987.pdf, with
        some modifications

        Version
        -------
        1.2.0

        Added GarNetRagged layer for more precise momentum direction estimation

        Date
        ----
        2023-11-02

        Parameters
        ----------
        Inputs : tuple(x ,rs)
            Ragged arrays containing the features of the detector hits in the form
                [eventNum x hits] x properties
            where "eventNum x hits" are separated by TensorFlow using the rowsplits
            rs.

        Returns
        -------
        Model(Inputs, Outputs)
            Inputs : same as Inputs \n
            Outputs : \n
            [p1, v1, p2, v2, sigma]
        """

        batchnorm_parameters = {
            "fluidity_decay": 0.1,
            "max_viscosity": 0.9999}

        x, rs = inputs
        rs = CastRowSplits()(rs)

        x = ScaledGooeyBatchNorm2(**batchnorm_parameters)(x)

        x_list = []
        for _ in range(3):
            x = RaggedGlobalExchange()([x, rs])
            x = tf.keras.layers.Dense(64, activation='elu')(x)
            x = tf.keras.layers.Dense(64, activation='elu')(x)

            x, *_ = RaggedGravNet(
                n_neighbours=200,
                n_dimensions=4,
                n_filters=64,
                n_propagate=64,
                feature_activation='elu'
            )([x, rs])
            x = ScaledGooeyBatchNorm2(**batchnorm_parameters)(x)
            x_list.append(x)

        x = tf.keras.layers.Concatenate(axis=1)(x_list)

        a_list = []
        for _ in range(5):
            x, a = GarNetRagged(
                n_aggregators=2,
                n_Fout_nodes=64,
                n_FLR_nodes=11
            )([x, rs])
            x = ScaledGooeyBatchNorm2(**batchnorm_parameters)(x)
            a_list.append(a)

        a = tf.keras.layers.Concatenate(axis=1)(a_list)
        a = tf.keras.layers.Flatten()(a)
        a = tf.keras.layers.Dense(256, activation='elu')(a)
        a = ScaledGooeyBatchNorm2(**batchnorm_parameters)(a)

        features = tf.keras.layers.Dense(12, activation="linear")(a)
        ln_sigma = tf.keras.layers.Dense(12, activation="relu")(a)
        outputs = tf.keras.layers.Concatenate(axis=1)([features, ln_sigma])
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def configure_training(self) -> (TrainingBase, simpleMetricsCallback):
        """ Set the model, optimizer and loss function for the training
        Also return metrics callbacks.
        """
        logger.info("Running training with the following parameters:")
        logger.info(f"Detector type: {self.detector_type}")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Train uncertainties: {self.train_uncertainties}")
        logger.info(f"Take weights: {self.takeweights}")

        assert self.detector_type and os.path.isdir(f"./nntr_data/{self.detector_type}"), \
            "Detector type not found in directory 'nntr_data'"
        os.makedirs(f"./nntr_models/{self.detector_type}/{self.model_name}", exist_ok=True)

        train = TrainingBase(
            inputDataCollection=f"./nntr_data/{self.detector_type}/Training/dataCollection.djcdc",
            outputDir=self.output_dir,
            valdata=f"./nntr_data/{self.detector_type}/Training/dataCollection.djcdc",
            takeweights=self.takeweights
        )

        if not train.modelSet():
            train.setModel(NNTR.two_vertex_fitter)
            train.saveCheckPoint("before_training.h5")
            train.setCustomOptimizer(tf.keras.optimizers.Adam())
            train.compileModel(
                learning_rate=1e-3,
                loss=nntr_L2_distance(train_uncertainties=self.train_uncertainties))
            train.keras_model.summary()

        cb = [
            simpleMetricsCallback(
                output_file=train.args["outputDir"]+'/losses.html',
                record_frequency=5,
                plot_frequency=5,
                select_metrics='*loss'),
            NanSweeper()]
        return train, cb

    def predict(self):
        Prediction(
            inputModel=f"{self.output_dir}/KERAS_check_best_model.h5",
            trainingDataCollection=f"{self.output_dir}/trainsamples.djcdc",
            inputSourceFileList=f"./nntr_data/{self.detector_type}/Testing/dataCollection.djcdc",
            outputDir=self.predicted_dir
        )
