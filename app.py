#TODO
"""
 - 1. Data Loader
 - 2. Model
 - 3. Convertor preds to video
"""

from libs.my.dataloader import DataLoader
from libs.my.core import Predictor, PredictorWithTiles
from libs.my.utils import MasksCreator
from libs.deeplab.model import Deeplabv3
from libs.ssd.vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor

video_path = './data/2.mp4'
model_path = './weights/mobilenet-v1-ssd-mp-0_675.pth'
weights_path = 'D:\\work\\MothionL\\weights\\mobilenet-v1-ssd-mp-0_675.pth'

if __name__ == '__main__':
    dataloader = DataLoader(video_path, 1)
    
    #net = create_mobilenetv1_ssd(21, is_test=True)
    #net.load(weights_path)
    #ssd = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    deeplab = Deeplabv3(backbone='xception', OS=8)
    
    
    #predictor = Predictor(ssd, deeplab)
    predictor = PredictorWithTiles(deeplab)
    preds = predictor.predict(dataloader)
    
    creator = MasksCreator(video_path, preds, './data/res2.avi')
    creator.cutter()

