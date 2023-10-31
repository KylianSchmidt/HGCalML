'''
Track Reconstruction algorithm for the simulation of the LUXE experiment
Uses the GravNet architecture
'''

import os
import tensorflow as tf
from Losses import nntr_L2_distance
from Layers import RaggedGlobalExchange
from RaggedLayers import CollapseRagged
from GravNetLayersRagged import (
    CastRowSplits, ScaledGooeyBatchNorm2, RaggedGravNet)
from GarNetRagged import GarNetRagged
from callbacks import NanSweeper
from training_base import TrainingBase
from predict import Prediction
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback


class NNTR():
    def __init__(
            self,
            detector_type="idealized_detector",
            model_name="v1_test",
            train_uncertainties=False,
            takeweights=""
            ):
        self.detector_type = detector_type
        self.model_name = model_name
        self.train_uncertainties = train_uncertainties
        self.takeweights = takeweights

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
        2023-10-24

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
        for _ in range(5):
            x = RaggedGlobalExchange()([x, rs])
            x = tf.keras.layers.Dense(64, activation='elu')(x)
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

        x_list = []
        for _ in range(10):
            x, *_ = GarNetRagged(
                n_aggregators=2,
                n_Fout_nodes=64,
                n_FLR_nodes=11
            )([x, rs])

        x = tf.keras.layers.Dense(512, activation='elu')(x)
        x = CollapseRagged('sum')([x, rs])
        x = ScaledGooeyBatchNorm2(**batchnorm_parameters)(x)

        features = tf.keras.layers.Dense(12, activation="linear")(x)
        ln_sigma = tf.keras.layers.Dense(12, activation="relu")(x)
        outputs = tf.keras.layers.Concatenate(axis=1)([features, ln_sigma])
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def configure_training(self) -> (TrainingBase, simpleMetricsCallback):
        """ Set the model, optimizer and loss function for the training
        Also return metrics callbacks.
        """
        assert self.detector_type and os.path.isdir(f"./nntr_data/{self.detector_type}"), \
            "Detector type not found in directory 'nntr_data'"
        os.makedirs(f"./nntr_models/{self.detector_type}/{self.model_name}", exist_ok=True)
        os.system(f"rm -rf ./nntr_models/{self.detector_type}/{self.model_name}/Output")

        train = TrainingBase(
            inputDataCollection=f"./nntr_data/{self.detector_type}/Training/dataCollection.djcdc",
            outputDir=f"./nntr_models/{self.detector_type}/{self.model_name}/Output",
            valdata=f"./nntr_data/{self.detector_type}/Training/dataCollection.djcdc"
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
            inputModel=f"nntr_models/{self.detector_type}/{self.model_name}/KERAS_check_best_model.h5",
            trainingDataCollection=f"./nntr_models/{self.detector_type}/{self.model_name}Output/trainsamples.djcdc",
            inputSourceFileList=f"./nntr_data/{self.detector_type}/Testing/Testing.djcdc",
            outputDir=f"./nntr_models/{self.detector_type}/{self.model_name}/Predicted"
        )


if __name__ == "__main__":
    """ Train the NNTR model from the Command Line
    """
    nntr = NNTR(
        train_uncertainties=False,
        detector_type="normal_detector",
        model_name="garnet/test_without_uncertainties"
    )

    train, cb = nntr.configure_training()
    train.change_learning_rate(5e-4)
    train.trainModel(
        nepochs=5,
        batchsize=10000,
        additional_callbacks=cb)

    train.change_learning_rate(1e-4)
    train.trainModel(
        nepochs=20,
        batchsize=10000,
        additional_callbacks=cb)

    train.change_learning_rate(1e-5)
    train.trainModel(
        nepochs=30,
        batchsize=10000,
        additional_callbacks=cb)

    nntr.predict()
