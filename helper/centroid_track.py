from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared = 80):
        self.maxDisappeared = maxDisappeared
        self.centroid       = OrderedDict()
        self.rects_stored   = OrderedDict()
        self.nextObjectID   = 0
    
    def register(self, centroid, rect):
        self.centroid[self.nextObjectID]     = centroid
        self.rects_stored[self.nextObjectID] = rect
        self.nextObjectID += 1
        
    def deregister(self, ID):
        del self.centroid[ID]
        del self.rects_stored[ID]
        
    def reset(self):
        self.centroid       = OrderedDict()
        self.rects_stored   = OrderedDict()
        
    def calculateCentroids(self, rects):
        centroids = []
        
        for i in range(0, len(rects)):
            
            (start_X, start_Y, end_X, end_Y) = rects[i]
            
            c_X = (start_X + end_X)/2
            c_Y = (start_Y + end_Y)/2
            
            centroids.append((c_X, c_Y))
        
        return centroids
        
        
    def update(self, input_rects):
        
        
        rect_centroid = self.calculateCentroids(input_rects)
        
        #update directly when previously no object detected
        if len(self.rects_stored) == 0:
            for i in range(0, len(input_rects)):
                self.register(rect_centroid[i], input_rects[i])
        
        #reset if currently no object detected
        elif len(input_rects) == 0:
                self.reset()
                
        #update old object with new detected object according to euclidean distance
        else:
            D                                    = dist.cdist(rect_centroid, list(self.centroid.values()))
            (total_new_object, total_old_object) = D.shape
            IDs                                  = list(self.centroid.keys())
            
            column_min = D.argmin(axis = 0)
            row_min    = D.argmin(axis = 1)
            
            #match old object with new object
            for i in range(0, total_new_object):
                
                centroid_index = row_min[i]
                
                if D[i, centroid_index] < self.maxDisappeared:
                    ID                    = IDs[centroid_index]
                    self.centroid[ID]     = rect_centroid[i]
                    self.rects_stored[ID] = input_rects[i]
                else:
                    self.register(rect_centroid[i], input_rects[i])
            
            #delete disappeared rectangle
            for i in range (0, total_old_object):
                
                centroid_index = i
                
                if D[column_min[i], i] >= self.maxDisappeared:
                    ID = IDs[centroid_index]
                    self.deregister(ID)
                    
            
            
if __name__ == "__main__":

    #Functions to aid testing*
    
    tracker = CentroidTracker()

    rect_a = [(20,20,60,60),(50,50,90,90)]
    print(tracker.calculateCentroids(rect_a))
    
    rect_b = [(100, 100, 150, 150), (150, 150, 200, 200)]
    print(tracker.calculateCentroids(rect_b))
    
    tracker.update(rect_a)
    print(tracker.centroid)
    
    tracker.update(rect_b)
    print(tracker.centroid)
    
    