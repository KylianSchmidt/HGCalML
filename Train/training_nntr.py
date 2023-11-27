'''
Track Reconstruction algorithm for the simulation of the LUXE experiment
Uses the GravNet architecture
'''

import os
import tensorflow as tf
from Losses import L2Distance, L2DistanceWithUncertainties
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
            train_uncertainties: bool,
            detector_type="idealized_detector",
            model_name="v1_test",
            takeweights="",
    ):
        self.detector_type = detector_type
        self.model_name = model_name
        self.train_uncertainties = train_uncertainties
        self.takeweights = takeweights
        self.output_dir = f"./nntr_models/{self.detector_type}/{self.model_name}/Output"
        self.predicted_dir = f"./nntr_models/{self.detector_type}/{self.model_name}/Predicted"

        if os.path.isdir(self.output_dir):
            print(
                "Output directory already exists, rename 'output_dir' to avoid overwriting")
        if os.path.isdir(self.predicted_dir):
            print(
                "Predicted directory already exists, rename 'predicted_dir' to avoid overwriting")

    def configure_training(
            self) -> (TrainingBase, simpleMetricsCallback):
        """ Set the model, optimizer and loss function for the training
        Also return metrics callbacks.
        """
        print("Running training with the following parameters:")
        print(f"Detector type: {self.detector_type}")
        print(f"Model name: {self.model_name}")
        print(f"Train uncertainties: {self.train_uncertainties}")
        print(f"Take weights: {self.takeweights}")

        assert self.detector_type and os.path.isdir(f"./nntr_data/{self.detector_type}"), \
            "Detector type not found in directory 'nntr_data'"
        os.makedirs(
            f"./nntr_models/{self.detector_type}/{self.model_name}", exist_ok=True)

        train = TrainingBase(
            inputDataCollection=f"./nntr_data/{self.detector_type}/Training/dataCollection.djcdc",
            outputDir=self.output_dir,
            valdata=f"./nntr_data/{self.detector_type}/Training/dataCollection.djcdc",
            takeweights=self.takeweights
        )

        if not train.modelSet():
            # Choose between training with or without uncertainties
            if self.train_uncertainties is True:
                model = NNTR.model_with_uncertainties
                loss = L2DistanceWithUncertainties()
            if self.train_uncertainties is False:
                model = NNTR.model_no_uncertainties
                loss = L2Distance()

            train.setModel(model)
            train.saveCheckPoint("before_training.h5")
            train.setCustomOptimizer(tf.keras.optimizers.Adam())
            train.compileModel(
                learning_rate=1e-3,
                loss=loss)
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
            outputDir=self.predicted_dir)

    def my_model(x, rs):
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
            [p1, v1, p2, v2]
        """
        batchnorm_parameters = {
            "fluidity_decay": 0.1,
            "max_viscosity": 0.9999}

        rs = CastRowSplits()(rs)

        x = ScaledGooeyBatchNorm2(**batchnorm_parameters)(x)
        x, a = GarNetRagged(
            n_aggregators=2,
            n_Fout_nodes=64,
            n_FLR_nodes=128
        )([x, rs])

        x_list = []
        for _ in range(5):
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
                n_FLR_nodes=128
            )([x, rs])
            x = ScaledGooeyBatchNorm2(**batchnorm_parameters)(x)
            a_list.append(a)

        a = tf.keras.layers.Concatenate(axis=1)(a_list)
        a = tf.keras.layers.Flatten()(a)
        a = tf.keras.layers.Dense(256, activation='elu')(a)
        a = ScaledGooeyBatchNorm2(**batchnorm_parameters)(a)

        return a

    def model_no_uncertainties(inputs):

        x, rs = inputs
        a = NNTR.my_model(x, rs)

        features = tf.keras.layers.Dense(12, activation="linear")(a)

        return tf.keras.Model(inputs, features)

    def model_with_uncertainties(inputs):

        x, rs = inputs
        a = NNTR.my_model(x, rs)

        features = tf.keras.layers.Dense(12, activation="linear")(a)
        ln_sigma = tf.keras.layers.Dense(12, activation="relu")(a)

        outputs = tf.keras.layers.Concatenate(axis=1)([features, ln_sigma])

        return tf.keras.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    """ Train the NNTR model from the Command Line

    NB: normal detector data without hits for calo being summed over has ~5000 hits per event
    normal detector with hits for calo being summed over has 100 hits per event
    idealized_detector has 102 hits per event
    """
    # Training
    nntr = NNTR(
        train_uncertainties=False,
        detector_type="normal_detector",
        model_name="garnet/11_27",
        # takeweights="./nntr_models/idealized_detector/garnet/11_21_v1/Output/KERAS_check_best_model.h5"
    )

    train, cb = nntr.configure_training()
    train.change_learning_rate(1e-3)
    train.trainModel(
        nepochs=2,
        batchsize=100000,
        additional_callbacks=cb)

    train.change_learning_rate(5e-4)
    train.trainModel(
        nepochs=5,
        batchsize=100000,
        additional_callbacks=cb)

    train.change_learning_rate(1e-4)
    train.trainModel(
        nepochs=20,
        batchsize=100000,
        additional_callbacks=cb)

    # Prediction
    print("Training finished, starting prediction")
    nntr.predict()
