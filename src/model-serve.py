#!/usr/bin/env python3
#

import sys
import argparse

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--topK", type=int, default=1, help="show the topK number of class predictions (default: 1)")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


def model_serve(args):
    # load the recognition network
    net = imageNet(args.network, sys.argv)

    # note: to hard-code the paths to load a model, the following API can be used:
    #
    # net = imageNet(model="model/resnet18.onnx", labels="model/labels.txt", 
    #                 input_blob="input_0", output_blob="output_0")

    # create video sources & outputs
    input = videoSource(args.input, argv=sys.argv)
    output = videoOutput(args.output, argv=sys.argv)
    font = cudaFont()

    # process frames until EOS or the user exits
    while True:
        # capture the next image
        img = input.Capture()

        if img is None: # timeout
            continue  

        # classify the image and get the topK predictions
        # if you only want the top class, you can simply run:
        #   class_id, confidence = net.Classify(img)
        predictions = net.Classify(img, topK=args.topK)

        # draw predicted class labels
        for n, (classID, confidence) in enumerate(predictions):
            classLabel = net.GetClassLabel(classID)
            confidence *= 100.0

            print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")

            font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}", 
                            x=5, y=5 + n * (font.GetSize() + 5),
                            color=font.White, background=font.Gray40)
                            
        # render the image
        output.Render(img)

        # update the title bar
        output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

        # print out performance info
        net.PrintProfilerTimes()

        # exit on input/output EOS
        if not input.IsStreaming() or not output.IsStreaming():
            break