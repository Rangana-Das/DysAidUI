from cv2 import cv2
import editdistance
from path import Path

from .DataLoaderIAM import DataLoaderIAM, Batch
from .Model import Model, DecoderType
from .SamplePreprocessor import preprocess

from .infer import segment
import os

from .WordSegmentation import wordSegmentation
from .WordSegmentation import prepareImg

from .Spelling import spelling

class FilePaths:
    
    #fnInfer = '../data/pragatitest.JPG'
    #fnCorpus = '../data/corpus.txt'
    def __init__(self, pathfn):
    	fnInfer=pathfn
    	"filenames and paths to data"
    	fnCharList = './model/charList.txt'
    	fnSummary = './model/summary.json'

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


def runmodel(filepath):
	model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
    result = infer(model, FilePaths.fnInfer)
    return result