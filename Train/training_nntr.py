'''
Track Reconstruction algorithm for the simulation of the LUXE experiment
Uses the GravNet architecture
'''

import os
import sys
import tensorflow as tf
from Losses import L2Distance, L2DistanceWithUncertainties, QuantileLoss
from Layers import RaggedGlobalExchange
from RaggedLayers import CollapseRagged
from GravNetLayersRagged import (
    CastRowSplits, ScaledGooeyBatchNorm2, RaggedGravNet)
from GarNetRagged import GarNetRagged
from callbacks import NanSweeper
from MetricsLayers import L2DistanceMetric
from training_base import TrainingBase
from predict import Prediction
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback


class NNTR():

    def __init__(
            self,
            model_type="with_uncertainties",
            detector_type="idealized_detector",
            model_name="v1_test",
            takeweights="",
            training_data_collection="",
            testing_data_collection="",
        ):
        self.detector_type = detector_type
        self.model_name = model_name
        self.model_type = model_type
        self.takeweights = takeweights
        self.training_data_collection = training_data_collection
        self.testing_data_collection = testing_data_collection
        self.output_dir = f"./nntr_models/{self.detector_type}/{self.model_name}/Output"
        self.predicted_dir = f"./nntr_models/{self.detector_type}/{self.model_name}/Predicted"
        if training_data_collection == "":
            self.training_data_collection = f"./nntr_data/{self.detector_type}/Training/dataCollection.djcdc"
        if testing_data_collection == "":
            self.testing_data_collection = f"./nntr_data/{self.detector_type}/Testing/dataCollection.djcdc"

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
        print(f"Model type: {self.model_type}")
        print(f"Take weights: {self.takeweights}")

        assert self.detector_type and os.path.isdir(f"./nntr_data/{self.detector_type}"), \
            "Detector type not found in directory 'nntr_data'"
        os.makedirs(
            f"./nntr_models/{self.detector_type}/{self.model_name}", exist_ok=True)

        train = TrainingBase(
            inputDataCollection=self.training_data_collection,
            outputDir=self.output_dir,
            valdata=self.training_data_collection,
            takeweights=self.takeweights
        )

        if not train.modelSet():
            # Choose between training with or without uncertainties
            if self.model_type == "with_uncertainties":
                model = NNTR.model_with_uncertainties
                loss = L2DistanceWithUncertainties()
            elif self.model_type == "no_uncertainties":
                model = NNTR.model_no_uncertainties
                loss = L2Distance()
            elif self.model_type == "quantile" or self.model_type == "quantiles":
                model = NNTR.model_with_quantiles
                loss = QuantileLoss()
            else:
                print("No such model type")
                return 0           

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
                select_metrics='*loss')]
        return train, cb

    def predict(self, epoch=""):
        Prediction(
            inputModel=f"{self.output_dir}/KERAS_check_best_model.h5",
            trainingDataCollection=f"{self.output_dir}/trainsamples.djcdc",
            inputSourceFileList=self.testing_data_collection,
            outputDir=self.predicted_dir+epoch)

    def my_model(x, rs):
        """ Network model for the BeamDumpTrackCalo two photon reconstruction

        Notes
        -----
        Specific Model Name:
        GravNet from the original paper https://arxiv.org/pdf/1902.07987.pdf, with
        some modifications

        Version
        -------
        1.3.0

        Added GarNetRagged layer for more precise momentum direction estimation

        Date
        ----
        2023-12-08

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

        x_list = []
        for _ in range(3):
            x = tf.keras.layers.Dense(64, activation='elu')(x)
            x = tf.keras.layers.Dense(64, activation='elu')(x)

            x, *_ = RaggedGravNet(
                n_neighbours=100,
                n_dimensions=4,
                n_filters=64,
                n_propagate=64,
                feature_activation='elu'
            )([x, rs])
            x = ScaledGooeyBatchNorm2(**batchnorm_parameters)(x)
            x_list.append(x)

        a_list = []
        for _ in range(3):
            x, a = GarNetRagged(
                n_aggregators=2,
                n_Fout_nodes=64,
                n_FLR_nodes=128
            )([x, rs])
            x = ScaledGooeyBatchNorm2(**batchnorm_parameters)(x)
            a_list.append(a)

        a = tf.keras.layers.Concatenate(axis=1)(a_list)
        a = tf.keras.layers.Flatten()(a)
        a = tf.keras.layers.Dense(128)(a)
        return a

    def model_no_uncertainties(inputs):

        x, rs = inputs
        a = NNTR.my_model(x, rs)

        features = tf.keras.layers.Dense(18, activation="linear")(a)

        return tf.keras.Model(inputs, features)

    def model_with_uncertainties(inputs):
        # Using ln(d/dx(gaussian(x, mu, sigma)))
        x, rs = inputs
        a = NNTR.my_model(x, rs)

        features = tf.keras.layers.Dense(18, activation="linear")(a)
        sigma = tf.keras.layers.Dense(
            18,
            activation="softplus",
            bias_initializer="ones",
            #activity_regularizer=tf.keras.regularizers.L2(1e-5),
            )(a)

        outputs = tf.keras.layers.Concatenate(axis=1)([features, sigma])

        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def model_with_quantiles(inputs):
        # Using quantile errors
        x, rs = inputs
        a = NNTR.my_model(x, rs)

        features = tf.keras.layers.Dense(18, activation="linear")(a)
        quantile_lower = tf.keras.layers.Dense(18, activation="linear")(a)
        quantile_upper = tf.keras.layers.Dense(18, activation="linear")(a)

        outputs = tf.keras.layers.Concatenate(axis=1)([features, quantile_lower, quantile_upper])

        return tf.keras.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    """ Train the NNTR model from the Command Line

    NB: normal detector data without hits for calo being summed over has ~1500 hits per event
    normal detector with hits for calo being summed over has 100 hits per event
    idealized_detector has 102 hits per event
    """
    if len(sys.argv) == 1:
    # Training
        nntr = NNTR(
            model_type="no_uncertainties",
            detector_type="normal_detector",
            model_name="comparison/12_30_nu_only_gravnet",
            #takeweights="./nntr_models/normal_detector/improved_gen/12_20_nu_v2/Output/KERAS_check_model_block_0_epoch_10.h5",
            training_data_collection="/ceph/kschmidt/beamdump/nntr_data/12_20_training/Training/dataCollection.djcdc",
            testing_data_collection="/ceph/kschmidt/beamdump/nntr_data/12_20_testing/Testing/dataCollection.djcdc"
        )
    elif len(sys.argv) == 4:
        nntr = NNTR(
            model_type=sys.argv[1],
            detector_type=sys.argv[2],
            model_name=sys.argv[3],
        )

    train, cb = nntr.configure_training()

    train.change_learning_rate(1e-3)
    train.trainModel(
        nepochs=10,
        batchsize=5000,
        additional_callbacks=cb)
    
    train.change_learning_rate(1e-4)
    train.trainModel(
        nepochs=40,
        batchsize=10000,
        additional_callbacks=cb)
    
    train.change_learning_rate(5e-5)
    train.trainModel(
        nepochs=50,
        batchsize=10000,
        additional_callbacks=cb)

    train.change_learning_rate(1e-5)
    train.trainModel(
        nepochs=100,
        batchsize=10000,
        additional_callbacks=cb)
    
    nntr.predict()
