import numpy as np
import cv2
from libs.deeplab.model import preprocess_input
from libs.my.tile import Tile

def calculate_boxes(boxes):
    # y1, x1, y2, x2
    # slide calculate
    #if boxes[1] > 1080-boxes[3]:
    #    slide = boxes[1]
    #else:
    #    slide = 1080-boxes[3]
    
    center = ((boxes[3]- boxes[1])/2 + boxes[1], (boxes[2]- boxes[0])/2 + boxes[0])
    return center #, slide

def cut_tile(frame, center, subsize):
    slide = subsize / 2
    tile = frame[int(center[0]-slide):int(center[0]+slide), int(center[1]-slide):int(center[1]+slide)]
    return tile

def batch_to_tiles(batch, boxes, batch_n, center_upper, center_bottom, subsize=512):
    upper_tiles = [] 
    bottom_tiles = []
    
    for i in range(0, batch.shape[0]):
        frame_n = i + batch.shape[0] * batch_n
        if (frame_n % 30 == 0) and (len(boxes[i]) > 0):
            center = calculate_boxes(boxes[i][0])
            center_upper = [center[0]-subsize, center[1]]
            center_bottom = [center[0]+subsize, center[1]]
            
            print("Upper:", center_upper)
            print("Bottom:", center_bottom)

            if center_upper[0] - subsize < 0:
                slide_up = center_upper[0] - subsize
                center_upper[0] -= (slide_up + 200)
            
            if center_bottom[0] + subsize > 1080:
                slide_up = 1080 - (center_bottom[0] + subsize)
                center_bottom[0] += (slide_up + 256)
        
        #upper_tiles.append(cut_tile(batch[i], center_upper, subsize))
        #bottom_tiles.append(cut_tile(batch[i], center_bottom, subsize))
        if i == 0:
            upper_batch = cut_tile(batch[i], center_upper, subsize)[None]
            bottom_batch = cut_tile(batch[i], center_bottom, subsize)[None]
        else:
            upper_batch = np.concatenate((upper_batch, cut_tile(batch[i], center_upper, subsize)[None]) , axis=0)
            bottom_batch = np.concatenate((bottom_batch, cut_tile(batch[i], center_bottom, subsize)[None]) , axis=0)
            
        
    return upper_batch, bottom_batch, center_upper, center_bottom

def tiles2masks(preds):
    for i in range(0, preds.shape[0]):
        if i == 0:
            res = np.argmax(preds[i].squeeze(), -1)[None]
        else:
            res1 = np.argmax(preds[i].squeeze(), -1)[None]
            res = np.concatenate((res, res1), axis=0)
    return res

def get_masks_batch(upper_masks, bottom_masks, center_upper, center_bottom, shape = (1080, 1920), slide = 256):
    for i in range(0, upper_masks.shape[0]):
        if i == 0:
            mask = np.zeros(shape, dtype=np.float64)
            mask[int(center_upper[0]-slide):int(center_upper[0]+slide), int(center_upper[1]-slide):int(center_upper[1]+slide)] = upper_masks[i]
            mask[int(center_bottom[0]-slide):int(center_bottom[0]+slide), int(center_bottom[1]-slide):int(center_bottom[1]+slide)] = bottom_masks[i]
            mask = mask[None]
        else:
            mask1 = np.zeros(shape, dtype=np.float64)
            mask1[int(center_upper[0]-slide):int(center_upper[0]+slide), int(center_upper[1]-slide):int(center_upper[1]+slide)] = upper_masks[i]
            mask1[int(center_bottom[0]-slide):int(center_bottom[0]+slide), int(center_bottom[1]-slide):int(center_bottom[1]+slide)] = bottom_masks[i]  
            mask = np.concatenate((mask, mask1[None]), axis=0)
    return mask       

class Predictor:
    def __init__(self, detector, net):
        self.detector = detector
        self.net = net
        #self.dataloader = None

    def predict(self, dataloader):
        """
            1. Получить батч
                1.1 Препроцессинг батч
            2. Задетектить боксы в батче
                2.1 Препроцессинг батч
            3. Взять маску
                3.1 Постпроцессинг батч
        """
        
        preds = []
        ret = True
        i = 0
        center_upper, center_bottom = None, None
        while ret:
            ret, batch = dataloader.__get_batch__()
            # boxes, labels, probs = self.detector.detect_batch(batch)
            boxes, labels, probs = self.__detect_batch(batch)
            upper_batch, bottom_batch, center_upper, center_bottom \
            = batch_to_tiles(batch, boxes, i, center_upper, center_bottom)
            
            upper_batch = preprocess_input(upper_batch)
            bottom_batch = preprocess_input(bottom_batch)
            
            upper_preds = self.net.predict(upper_batch)
            bottom_preds = self.net.predict(bottom_batch)
            
            upper_masks = tiles2masks(upper_preds)
            bottom_masks = tiles2masks(bottom_preds)

            preds_batch = get_masks_batch(upper_masks, bottom_masks, center_upper, center_bottom)

            for k in range(0, preds_batch.shape[0]):
                preds.append(preds_batch[k])

            i += 1
        return preds
            
    def __detect_batch(self, batch):
        boxes_ = []
        labels_ = []
        probs_ = []
        
        for i in range(0, batch.shape[0]):
            image = cv2.cvtColor(batch[i], cv2.COLOR_BGR2RGB)
            boxes, labels, probs = self.detector.predict(image, 10, 0.4)
            boxes_.append(boxes)
            labels_.append(labels)
            probs_.append(probs)
            
        return boxes_, labels_, probs_

class PredictorWithTiles:
    def __init__(self, net):
        self.net = net

    def predict(self, dataloader):
        """
            1. Получить батч
                1.1 Препроцессинг батч
            2. Задетектить боксы в батче
                2.1 Препроцессинг батч
            3. Взять маску
                3.1 Постпроцессинг батч
        """
        
        preds = []
        mask = None
        ret = True
        i = 0
        while ret:
            print(f"Processing batch {i}")
            ret, batch = dataloader.__get_batch__()
            if ret:
                tiles, length, coords = self.__batch_to_tiles(batch, i)
                tiles_process = preprocess_input(tiles)
                batch = None
                for k in range(0, 15, 3):
                    tiles_batch = tiles_process[k:3+k]
                    if batch is None:
                       batch = self.net.predict(tiles_batch)
                    else:
                        batch1 = self.net.predict(tiles_batch)
                        batch = np.concatenate([batch, batch1], axis=0)

                masks_preds = tiles2masks(batch)
                masks = self.__tiles2mask(masks_preds, length, coords)
                preds.append(masks)
                i += 1

        for pred in preds:
            if mask is None:
                mask = pred
            else:
                mask = np.concatenate([mask, pred], axis = 0)

        return mask

    def __batch_to_tiles(self, batch, batch_n):
        #return [Tile.split_frame(batch[i], i + batch.shape[0] * batch_n) for i in range(0, batch.shape[0])]
        tiles = [Tile.split_frame(batch[i], i + batch.shape[0] * batch_n) for i in range(0, batch.shape[0])]
        coords = []
        length = len(tiles[0])
        tilesOut = None
        for tilelist in tiles:
            for tile in tilelist:
                if tilesOut is None:
                    tilesOut = tile.roi[None]
                    coords.append((tile.up, tile.left))
                else:
                    tilesOut = np.concatenate([tilesOut, tile.roi[None]], axis=0)
                    coords.append((tile.up, tile.left))
        print("length:", length)
        return tilesOut, length, coords

    def __tiles2mask(self, tilesIn, length, coords, shape=(1080, 1920)):
        bs = len(tilesIn) // length
        print(bs)
        predictBatch = []
        for i in range(0, bs):
            for k in range(length*i, length*(i+1)):
                image = np.zeros(shape, dtype=np.uint8)
                image[coords[k][0]:coords[k][0]+512,coords[k][1]:coords[k][1]+512] = tilesIn[k]
                predictBatch.append(image)
        return predictBatch
