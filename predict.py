#!/usr/bin/env python3

import os
import logging
from argparse import ArgumentParser
import tensorflow as tf
from DeepJetCore.training.gpuTools import DJCSetGPUs
from DeepJetCore.customObjects import get_custom_objects
from DeepJetCore.dataPipeline import TrainDataGenerator
from DeepJetCore.DataCollection import DataCollection

custom_objs = get_custom_objects()
logger = logging.getLogger(__name__)


class Prediction():
    def __init__(
        self,
        inputModel: str,
        trainingDataCollection: str,
        inputSourceFileList: str,
        outputDir: str,
        batchsize: int = -1,
        gpu: str = "",
        unbuffered: bool = False,
        pad_rowsplits: bool = False
    ):

        self.args = {
            "inputModel": inputModel,
            "trainingDataCollection": trainingDataCollection,
            "inputSourceFileList": inputSourceFileList,
            "outputDir": outputDir,
            "batchsize": batchsize,
            "gpu": gpu,
            "unbuffered": unbuffered,
            "pad_rowsplits": pad_rowsplits
        }

        inputdatafiles = []
        inputdir = None
        DJCSetGPUs(self.args["gpu"])
        model = tf.keras.models.load_model(
            self.args["inputModel"],
            custom_objects=custom_objs)

        # Prepare input lists for different file formats
        if self.args["inputSourceFileList"][-6:] == ".djcdc":
            logger.info('Reading from data collection', self.args["inputSourceFileList"])
            predsamples = DataCollection(self.args["inputSourceFileList"])
            inputdir = predsamples.dataDir

            for s in predsamples.samples:
                inputdatafiles.append(s)

        elif self.args["inputSourceFileList"][-6:] == ".djctd":
            inputdir = os.path.abspath(os.path.dirname(self.args["inputSourceFileList"]))
            infile = os.path.basename(self.args["inputSourceFileList"])
            inputdatafiles.append(infile)

        else:
            logger.info('Reading from text file', self.args["inputSourceFileList"])
            inputdir = os.path.abspath(os.path.dirname(self.args["inputSourceFileList"]))

            with open(self.args["inputSourceFileList"], "r") as f:
                for s in f:
                    inputdatafiles.append(s.replace('\n', '').replace(" ", ""))

        dc = None
        if (self.args["inputSourceFileList"][-6:] == ".djcdc" and
                not self.args["trainingDataCollection"][-6:] == ".djcdc"):
            dc = DataCollection(self.args["inputSourceFileList"])
            
            if self.args["batchsize"] < 1:
                batchsize = 1

            logger.info('No training data collection given. Using batch size of', batchsize)
        else:
            dc = DataCollection(self.args["trainingDataCollection"])

        outputs = []
        os.system('mkdir -p ' + self.args["outputDir"])

        for inputfile in inputdatafiles:
            logger.info(f"Predicting {inputdir}/{inputfile}")
            use_inputdir = inputdir

            if inputfile[0] == "/":
                use_inputdir = ""

            outfilename = "pred_" + os.path.basename(inputfile)
            td = dc.dataclass()

            if inputfile[-5:] == 'djctd':
                if self.args["unbuffered"]:
                    td.readFromFile(use_inputdir+"/"+inputfile)
                else:
                    td.readFromFileBuffered(use_inputdir+"/"+inputfile)
            else:
                logger.info('Converting '+inputfile)
                td.readFromSourceFile(
                    f"{use_inputdir}/{inputfile}",
                    dc.weighterobjects,
                    istraining=False)

            if self.args["batchsize"] < 1:
                batchsize = dc.getBatchSize()

            logger.info(f'Batch size = {batchsize}')
            gen = TrainDataGenerator()
            gen.setBatchSize(batchsize)
            gen.setSquaredElementsLimit(dc.batch_uses_sum_of_squares)
            gen.setSkipTooLargeBatches(False)
            gen.setBuffer(td)
            predicted = model.predict(
                gen.feedNumpyData(),
                steps=gen.getNBatches(),
                max_queue_size=1,
                use_multiprocessing=False, verbose=1)

            features = td.transferFeatureListToNumpy(self.args["pad_rowsplits"])
            truth = td.transferTruthListToNumpy(self.args["pad_rowsplits"])
            weights = td.transferWeightListToNumpy(self.args["pad_rowsplits"])
            td.clear()
            gen.clear()

            # Circumvent that keras return only an array if there is just one list item
            if not isinstance(predicted, list):
                predicted = [predicted]

            overwrite_outname = td.writeOutPrediction(
                predicted=predicted,
                features=features,
                truth=truth,
                weights=weights,
                outfilename=f'{self.args["outputDir"]}/{outfilename}',
                inputfile=f"{use_inputdir}/{inputfile}")

            if overwrite_outname is not None:
                outfilename = overwrite_outname
            outputs.append(outfilename)

        with open(f'{self.args["outputDir"]}/outfiles.txt', "w") as f:
            for output in outputs:
                f.write(f"{output}\n")


if __name__ == "__main__":
    parser = ArgumentParser('Apply a model to a (test) source sample.')
    parser.add_argument('inputModel')
    parser.add_argument(
        'trainingDataCollection',
        help="the training data collection. Used to infer data format and batch size.")
    parser.add_argument(
        "inputSourceFileList",
        help="can be text file or a DataCollection file in the same directory as the sample files, or just a single traindata file.")
    parser.add_argument(
        'outputDir',
        help="will be created if it doesn't exist.")
    parser.add_argument(
        "--batchsize",
        type=int,
        help="batch size, overrides the batch size from the training data collection.",
        default=-1)
    parser.add_argument(
        "--gpu",
        help="select specific GPU",
        metavar="OPT",
        default="")
    parser.add_argument(
        "--unbuffered",
        help="do not read input in memory buffered mode (for lower memory consumption on fast disks)",
        default=False,
        action="store_true")
    parser.add_argument(
        "--pad_rowsplits",
        help="pad the row splits if the input is ragged",
        default=False,
        action="store_true")
    Prediction(**vars(parser.parse_args()))
