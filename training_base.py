import sys
import os
import shutil
import tensorflow as tf
import copy
import matplotlib
import logging
from _thread import start_new_thread
from DeepJetCore.training.tokenTools import checkTokens, renew_token_process
from DeepJetCore.training.gpuTools import DJCSetGPUs
from DeepJetCore import DataCollection
from DeepJetCore.customObjects import get_custom_objects
from DeepJetCore.modeltools import apply_weights_where_possible, load_model
from DeepJetCore.training.batchTools import submit_batch
from DeepJetCore.training.DeepJet_callbacks import DeepJet_callbacks

matplotlib.use('Agg')
custom_objects_list = get_custom_objects()
logger = logging.getLogger(__name__)


class TrainingBase(object):

    def __init__(
            self,
            inputDataCollection: str,
            outputDir: str,
            valdata: str = "",
            takeweights: str = "",
            split_train_and_test=0.85,
            use_weights=False,
            test_run=False,
            test_run_fraction=0.1,
            renew_tokens=False,
            collection_class=DataCollection,
            resume_silently=False,
            recreate_silently=False
    ):

        scriptname = sys.argv[0]
        self.argstring = sys.argv
        self.args = {
            "inputDataCollection": inputDataCollection,
            "outputDir": outputDir,
            "modelMethod": None,
            "gpu": "",
            "gpufraction": -1,
            "submitbatch": False,
            "walltime": '1d',
            "isbatchrun": False,
            "valdata": valdata,
            "takeweights": takeweights
        }

        if self.args["isbatchrun"]:
            self.args["submitbatch"] = False
            resume_silently = True

        if self.args["submitbatch"]:
            logger.info('Submitting batch job. Model will be compiled for testing before submission '
                        + '(GPU settings being ignored)')

        DJCSetGPUs(self.args["gpu"])

        if self.args["gpufraction"] > 0 and self.args["gpufraction"] < 1:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=self.args["gpufraction"])
            session = tf.Session(
                config=tf.ConfigProto(gpu_options=gpu_options))
            tf.keras.backend.set_session(session)
            logger.info(
                f"Using gpu memory fraction: {self.args['gpufraction']}")

        self.ngpus = 1
        self.dist_strat_scope = None

        if len(self.args["gpu"]):
            self.ngpus = len([i for i in self.args["gpu"].split(',')])
            logger.info('Running on ' + str(self.ngpus) + ' gpus')

            if self.ngpus > 1:
                self.dist_strat_scope = tf.distribute.MirroredStrategy()

        self.keras_inputs = []
        self.keras_inputsshapes = []
        self.keras_model = None
        self.keras_model_method = self.args["modelMethod"]
        self.keras_weight_model_path = self.args["takeweights"]
        self.train_data = None
        self.val_data = None
        self.start_learning_rate = None
        self.optimizer = None
        self.trainedepoches = 0
        self.compiled = False
        self.checkpointcounter = 0
        self.renew_tokens = renew_tokens
        if self.args["isbatchrun"]:
            self.renew_tokens = False
        self.callbacks = None
        self.custom_optimizer = False
        self.copied_script = ""
        self.submitbatch = self.args["submitbatch"]
        self.GAN_mode = False
        isNewTraining = True

        if ',' not in self.args["inputDataCollection"]:
            self.inputData = os.path.abspath(self.args["inputDataCollection"])
        else:
            self.inputData = [os.path.abspath(
                i) for i in self.args["inputDataCollection"].split(',')]

        if os.path.isdir(self.args["outputDir"]):
            if not (resume_silently or recreate_silently):
                var = input(
                    'Output dir exists. To recover a training, please type "yes"\n')
                if not var == 'yes':
                    raise Exception('output directory must not exists yet')

            isNewTraining = False

            if recreate_silently:
                isNewTraining = True
        else:
            os.mkdir(self.args["outputDir"])

        self.args["outputDir"] = os.path.abspath(self.args["outputDir"])
        self.args["outputDir"] += '/'

        if recreate_silently:
            os.system('rm -rf ' + self.args["outputDir"] + '*')

        # Copy configuration to output dir
        if not self.args["isbatchrun"]:
            try:
                shutil.copyfile(
                    scriptname, self.args["outputDir"]+os.path.basename(scriptname))
            except shutil.SameFileError:
                pass
            except BaseException as e:
                raise e

            self.copied_script = self.args["outputDir"] + \
                os.path.basename(scriptname)
        else:
            self.copied_script = scriptname

        self.train_data = collection_class()
        self.train_data.readFromFile(self.inputData)
        self.train_data.use_weights = use_weights

        if len(self.args["valdata"]):
            logger.info('using validation data from ', self.args["valdata"])
            self.val_data = DataCollection(self.args["valdata"])

        else:
            if test_run:
                if len(self.train_data) > 1:
                    self.train_data.split(test_run_fraction)

                self.train_data.dataclass_instance = None  # can't be pickled
                self.val_data = copy.deepcopy(self.train_data)

            else:
                self.val_data = self.train_data.split(split_train_and_test)

        shapes = self.train_data.getNumpyFeatureShapes()
        inputdtypes = self.train_data.getNumpyFeatureDTypes()
        inputnames = self.train_data.getNumpyFeatureArrayNames()

        for i in range(len(inputnames)):
            if not inputnames[i] or inputnames[i] == "_rowsplits":
                inputnames[i] = f"input_{i}{inputnames[i]}"

        logger.info("shapes", shapes)
        logger.info("inputdtypes", inputdtypes)
        logger.info("inputnames", inputnames)

        self.keras_inputs = []
        self.keras_inputsshapes = []

        for s, dt, n in zip(shapes, inputdtypes, inputnames):
            self.keras_inputs.append(
                tf.keras.layers.Input(shape=s, dtype=dt, name=n))
            self.keras_inputsshapes.append(s)

        self.train_data.writeToFile(
            self.args["outputDir"]+'trainsamples.djcdc', abspath=True)
        self.val_data.writeToFile(
            self.args["outputDir"]+'valsamples.djcdc', abspath=True)

        if not isNewTraining:
            kfile = self.args["outputDir"]+'/KERAS_check_model_last.h5'
            if not os.path.isfile(kfile):
                kfile = self.args["outputDir"] + \
                    '/KERAS_check_model_last'  # savedmodel format
                if not os.path.isdir(kfile):
                    kfile = ''

            if len(kfile):
                logger.info('loading model', kfile)

                if self.dist_strat_scope is not None:
                    with self.dist_strat_scope.scope():
                        self.loadModel(kfile)
                else:
                    self.loadModel(kfile)
                self.trainedepoches = 0

                if os.path.isfile(self.args["outputDir"]+'losses.log'):
                    for line in open(self.args["outputDir"]+'losses.log'):
                        valloss = line.split(' ')[1][:-1]

                        if not valloss == "None":
                            self.trainedepoches += 1
                else:
                    logger.info(
                        'incomplete epochs, starting from the beginning but with pretrained model')
            else:
                logger.info(
                    'no model found in existing output dir, starting training from scratch')

    def __del__(self):
        if hasattr(self, 'train_data'):
            del self.train_data
            del self.val_data

    def modelSet(self):
        return (self.keras_model is not None) and (self.keras_weight_model_path is not None)

    def setDJCKerasModel(self, model, *args, **kwargs):

        if len(self.keras_inputs) < 1:
            raise Exception('setup data first')

        self.keras_model = model(*args, **kwargs)

        if hasattr(self.keras_model, "_is_djc_keras_model"):
            self.keras_model.setInputShape(self.keras_inputs)
            self.keras_model.build(None)

        if not self.keras_model:
            raise Exception('Setting DJCKerasModel not successful')

    def setModel(self, model, **modelargs):
        if len(self.keras_inputs) < 1:
            raise Exception('setup data first')

        if self.dist_strat_scope is not None:
            with self.dist_strat_scope.scope():
                self.keras_model = model(self.keras_inputs, **modelargs)
        else:
            self.keras_model = model(self.keras_inputs, **modelargs)
        if hasattr(self.keras_model, "_is_djc_keras_model"):
            self.keras_model.setInputShape(self.keras_inputs)
            self.keras_model.build(None)

        if len(self.keras_weight_model_path):
            self.keras_model = apply_weights_where_possible(
                self.keras_model,
                load_model(self.keras_weight_model_path))

        if not self.keras_model:
            raise Exception('Setting model not successful')

    def saveCheckPoint(self, addstring=''):
        self.checkpointcounter = self.checkpointcounter + 1
        self.saveModel(
            f"KERAS_model_checkpoint_{self.checkpointcounter}_{addstring}")

    def _loadModel(self, filename):
        keras_model = tf.keras.models.load_model(
            filename, custom_objects=custom_objects_list)
        return keras_model, keras_model.optimizer

    def loadModel(self, filename):
        self.keras_model, self.optimizer = self._loadModel(filename)
        self.compiled = True

        if self.ngpus > 1:
            self.compiled = False

    def setCustomOptimizer(self, optimizer):
        self.optimizer = optimizer
        self.custom_optimizer = True

    def compileModel(
            self,
            learning_rate,
            clipnorm=None,
            print_models=False,
            metrics=None,
            is_eager=False,
            **compile_args
    ):
        if not self.keras_model and not self.GAN_mode:
            raise Exception('set model first')

        if self.ngpus > 1 and not self.submitbatch:
            logger.info('Model being compiled for '+str(self.ngpus)+' gpus')

        self.start_learning_rate = learning_rate

        if not self.custom_optimizer:
            if clipnorm:
                self.optimizer = tf.keras.optimizers.Adam(
                    lr=self.start_learning_rate, clipnorm=clipnorm)
            else:
                self.optimizer = tf.keras.optimizers.Adam(
                    lr=self.start_learning_rate)

        if self.dist_strat_scope is not None:
            with self.dist_strat_scope.scope():
                self.keras_model.compile(
                    optimizer=self.optimizer, metrics=metrics, **compile_args)
        else:
            self.keras_model.compile(
                optimizer=self.optimizer, metrics=metrics, **compile_args)

        if is_eager:
            # call on one batch to fully build it
            self.keras_model(self.train_data.getExampleFeatureBatch())

        if print_models:
            logger.info(self.keras_model.summary())
        self.compiled = True

    def compileModelWithCustomOptimizer(
            self,
            customOptimizer,
            **compile_args):
        raise Exception(
            'DEPRECATED: please use setCustomOptimizer before calling compileModel')

    def saveModel(self, outfile):
        if not self.GAN_mode:
            self.keras_model.save(self.args["outputDir"]+outfile)
        else:
            self.gan.save(self.args["outputDir"]+'GAN_'+outfile)
            self.generator.save(self.args["outputDir"]+'GEN_'+outfile)
            self.discriminator.save(self.args["outputDir"]+'DIS_'+outfile)

    def _initTraining(self, nepochs, batchsize, use_sum_of_squares=False):
        if self.submitbatch:
            submit_batch(self, self.args["walltime"])
            exit()  # don't delete this!

        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)
        self.train_data.batch_uses_sum_of_squares = use_sum_of_squares
        self.val_data.batch_uses_sum_of_squares = use_sum_of_squares

        # make sure tokens don't expire
        if self.renew_tokens:
            logger.info(
                'afs backgrounder has proven to be unreliable, use with care')
            checkTokens()
            start_new_thread(renew_token_process, ())

        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)

    def trainModel(
            self,
            nepochs,
            batchsize,
            run_eagerly=False,
            batchsize_use_sum_of_squares=False,
            fake_truth=False,
            stop_patience=-1,
            lr_factor=0.5,
            lr_patience=-1,
            lr_epsilon=0.003,
            lr_cooldown=6,
            lr_minimum=0.000001,
            checkperiod=10,
            backup_after_batches=-1,
            additional_plots=None,
            additional_callbacks=None,
            load_in_mem=False,
            max_files=-1,
            plot_batch_loss=False,
            **trainargs
    ):
        self.keras_model.run_eagerly = run_eagerly
        # write only after the output classes have been added
        self._initTraining(nepochs, batchsize, batchsize_use_sum_of_squares)
        # won't work for purely eager models
        self.keras_model.save(self.args["outputDir"]+'KERAS_untrained_model')

        logger.info('setting up callbacks')
        minTokenLifetime = 5
        if not self.renew_tokens:
            minTokenLifetime = -1

        self.callbacks = DeepJet_callbacks(
            self.keras_model,
            stop_patience=stop_patience,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_epsilon=lr_epsilon,
            lr_cooldown=lr_cooldown,
            lr_minimum=lr_minimum,
            outputDir=self.args["outputDir"],
            checkperiod=checkperiod,
            backup_after_batches=backup_after_batches,
            checkperiodoffset=self.trainedepoches,
            additional_plots=additional_plots,
            batch_loss=plot_batch_loss,
            print_summary_after_first_batch=run_eagerly,
            minTokenLifetime=minTokenLifetime)

        if additional_callbacks is not None:
            if not isinstance(additional_callbacks, list):
                additional_callbacks = [additional_callbacks]
            self.callbacks.callbacks.extend(additional_callbacks)

        logger.info('starting training')

        if load_in_mem:
            logger.info('make features')
            X_train = self.train_data.getAllFeatures(nfiles=max_files)
            X_test = self.val_data.getAllFeatures(nfiles=max_files)
            logger.info('make truth')
            Y_train = self.train_data.getAllLabels(nfiles=max_files)
            Y_test = self.val_data.getAllLabels(nfiles=max_files)
            self.keras_model.fit(
                X_train,
                Y_train,
                batch_size=batchsize,
                epochs=nepochs,
                callbacks=self.callbacks.callbacks,
                validation_data=(X_test, Y_test),
                max_queue_size=1,
                use_multiprocessing=False,
                workers=0,
                **trainargs)
        else:
            # prepare generator
            logger.info("setting up generator... can take a while")
            use_fake_truth = None

            if fake_truth:
                if isinstance(self.keras_model.output, dict):
                    use_fake_truth = [
                        k for k in self.keras_model.output.keys()]
                elif isinstance(self.keras_model.output, list):
                    use_fake_truth = len(self.keras_model.output)

            traingen = self.train_data.invokeGenerator(
                fake_truth=use_fake_truth)
            valgen = self.val_data.invokeGenerator(fake_truth=use_fake_truth)

            while (self.trainedepoches < nepochs):
                # this can change from epoch to epoch
                # calculate steps for this epoch
                # feed info below
                traingen.prepareNextEpoch()
                valgen.prepareNextEpoch()
                nbatches_train = traingen.getNBatches()  # might have changed due to shuffeling
                nbatches_val = valgen.getNBatches()

                logger.info('>>>> epoch', self.trainedepoches, "/", nepochs)
                logger.info('training batches: ', nbatches_train)
                logger.info('validation batches: ', nbatches_val)

                self.keras_model.fit(
                    traingen.feedNumpyData(),
                    steps_per_epoch=nbatches_train,
                    epochs=self.trainedepoches + 1,
                    initial_epoch=self.trainedepoches,
                    callbacks=self.callbacks.callbacks,
                    validation_data=valgen.feedNumpyData(),
                    validation_steps=nbatches_val,
                    max_queue_size=1,
                    use_multiprocessing=False,
                    workers=0,
                    **trainargs
                )
                self.trainedepoches += 1
                traingen.shuffleFileList()

            self.saveModel("KERAS_model.h5")

        return self.keras_model, self.callbacks.history

    def change_learning_rate(self, new_lr):
        if self.GAN_mode:
            tf.keras.backend.set_value(self.discriminator.optimizer.lr, new_lr)
            tf.keras.backend.set_value(self.gan.optimizer.lr, new_lr)
        else:
            tf.keras.backend.set_value(
                self.keras_model.optimizer.lr, new_lr)
