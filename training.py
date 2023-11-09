from Train.training_nntr import NNTR


if __name__ == "__main__":
    """ Train the NNTR model from the Command Line
    """
    nntr = NNTR(
        train_uncertainties=True,
        detector_type="idealized_detector",
        model_name="garnet/v1_with_uncertainties",
        #takeweights="./nntr_models/idealized_detector/garnet/test_with_uncertainties/Output/KERAS_check_best_model.h5"
    )
    print("Model name:", nntr.model_name)
    print("Detector type:", nntr.detector_type)
    print("Train uncertainties:", nntr.train_uncertainties)

    train, cb = nntr.configure_training()
    train.change_learning_rate(1e-3)
    train.trainModel(
        nepochs=1,
        batchsize=1024,
        additional_callbacks=cb)

    train.change_learning_rate(5e-4)
    train.trainModel(
        nepochs=5,
        batchsize=1024,
        additional_callbacks=cb)

    train.change_learning_rate(1e-4)
    train.trainModel(
        nepochs=10,
        batchsize=1024,
        additional_callbacks=cb)

    train.change_learning_rate(1e-5)
    train.trainModel(
        nepochs=20,
        batchsize=1024,
        additional_callbacks=cb)

    print("Training finished, starting prediction")
    nntr.predict()
