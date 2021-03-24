import argparse
import json

from cv2 import cv2
import editdistance
from path import Path

from .DataLoaderIAM import DataLoaderIAM, Batch
from Model import Model, DecoderType
from .SamplePreprocessor import preprocess

from infer import segment
import os

from WordSegmentation import wordSegmentation
from WordSegmentation import prepareImg

from Spelling import spelling

class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnSummary = '../model/summary.json'
    fnInfer = '../data/pragatitest.JPG'
    fnCorpus = '../data/corpus.txt'

#test for integration
def test_integration():
    text=["This is a test","Should not be printed"]
    return text[0]

def write_summary(charErrorRates, wordAccuracies):
    with open(FilePaths.fnSummary, 'w') as f:
        json.dump({'charErrorRates': charErrorRates, 'wordAccuracies': wordAccuracies}, f)


def train(model, loader):
    "train NN"
    epoch = 0  # number of training epochs since start
    summaryCharErrorRates = []
    summaryWordAccuracies = []
    bestCharErrorRate = float('inf')  # best valdiation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occured
    earlyStopping = 25  # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print(f'Epoch: {epoch} Batch: {iterInfo[0]}/{iterInfo[1]} Loss: {loss}')

        # validate
        charErrorRate, wordAccuracy = validate(model, loader)

        # write summary
        summaryCharErrorRates.append(charErrorRate)
        summaryWordAccuracies.append(wordAccuracy)
        write_summary(summaryCharErrorRates, summaryWordAccuracies)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {charErrorRate * 100.0}%')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print(f'No more improvement since {earlyStopping} epochs. Training stopped.')
            break


def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print(f'Batch: {iterInfo[0]} / {iterInfo[1]}')
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print(f'Character error rate: {charErrorRate * 100.0}%. Word accuracy: {wordAccuracy * 100.0}%.')
    return charErrorRate, wordAccuracy


def fetchforinfer(pathimg):
    #return segment(pathimg)
    img = prepareImg(cv2.imread(pathimg), 50)
    res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)

    print('Segmented into %d words'%len(res))

    words=[]
    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
        (x, y, w, h) = wordBox
        words.append(wordImg)
        #cv2.imwrite('../out/pg.jpg/%d.png'%(j), wordImg) # save word
        #cv2.rectangle(img,(x,y),(x+w,y+h),0,1) 
    return [words]
    

def infer(model, fnImg):
    "recognize text in image provided by file path"

    outF = open("../data/recog.txt", "w")

    #img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    #img2= preprocess(cv2.imread("../data/2.png", cv2.IMREAD_GRAYSCALE), Model.imgSize)

    pagewiselist=fetchforinfer(fnImg)
    #print(len(pagewiselist))
    #print(type(pagewiselist[0]))
    for pagesin in range(len(pagewiselist)):
        for (i,j) in enumerate(pagewiselist[pagesin]):
            pagewiselist[pagesin][i]=preprocess(j,Model.imgSize)
        batch = Batch(None, pagewiselist[pagesin])
        (recognized, probability) = model.inferBatch(batch, True)

        for i in range(len(recognized)):
            outF.write(recognized[i])
            outF.write(" ")
            print("Recognized:",recognized[i]," ; Probability:",probability[i])
        outF.write('\n')
    
    outF.close()

    s=spelling()
    s.correct()
    corrected=s.punct()
    return corrected[0]
    #print(corrected[0])
    #outF1 = open("../data/corrected.txt", "w+")
    #outF1.write(corrected[0])
    #outF1.close()

    #print(f'Recognized: "{recognized[0]}","{recognized[1]}"')
    #print(f'Probability: {probability[0]},{probability[1]}')


def main():
    "main function"
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath',
                        help='CTC decoder')
    parser.add_argument('--batch_size', help='batch size', type=int, default=100)
    parser.add_argument('--data_dir', help='directory containing IAM dataset', type=Path, required=False)
    parser.add_argument('--fast', help='use lmdb to load images', action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')
    args = parser.parse_args()

    # set chosen CTC decoder
    if args.decoder == 'bestpath':
        decoderType = DecoderType.BestPath
    elif args.decoder == 'beamsearch':
        decoderType = DecoderType.BeamSearch
    elif args.decoder == 'wordbeamsearch':
        decoderType = DecoderType.WordBeamSearch

    # train or validate on IAM dataset
    if args.train or args.validate:
        # load training data, create TF model
        loader = DataLoaderIAM(args.data_dir, args.batch_size, Model.imgSize, Model.maxTextLen, args.fast)

        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

        # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

        # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)

    # infer text on test image
    else:
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
        infer(model, FilePaths.fnInfer)


if __name__ == '__main__':
    main()
