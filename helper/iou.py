def eliminate_overlap(rectangles, confidences, threshold = 0.5):
    
    rect_length  = len(rectangles)
    overlap_list = find_overlap_rectangle_index(rectangles, confidences, threshold = 0.5)
    
    #select rectangle that is not label as overlap rectangle
    final_list = [rectangles[i] for i in range(0, rect_length) if i not in overlap_list]
    
    return final_list


def calculate_iou_of(rect_target, rect_b):
    
    (x_tar, y_tar, x_end_tar, y_end_tar) = rect_target
    (x_b  , y_b  , x_end_b  , y_end_b  ) = rect_b
    
    overlap_x     = max(x_tar, x_b)
    overlap_y     = max(y_tar, y_b)
    overlap_end_x = min(x_end_tar, x_end_b)
    overlap_end_y = min(y_end_tar, y_end_b)
    
    area_of_target = (x_end_tar     - x_tar    ) * (y_end_tar     - y_tar    )
    area_of_overlap= (overlap_end_x - overlap_x) * (overlap_end_y - overlap_y)
    
    iou = area_of_overlap/area_of_target
    
    return iou
    
def check_overlap(rect_target, rect_b):
    (x_tar, y_tar, x_end_tar, y_end_tar) = rect_target
    (x_b  , y_b  , x_end_b  , y_end_b  ) = rect_b
    
    isOverlap = True
    
    #check rect b at left/right of rect target with no overlap
    if x_b > x_end_tar or x_end_b < x_tar:
        isOverlap = False
    
    #check rect b at top/bottom of rect target with no overlap
    if y_b > y_end_tar or y_end_b < y_tar:
        isOverlap = False
        
    return isOverlap

def find_overlap_rectangle_index(rectangles, confidences, threshold = 0.5):
    
    overlap_list       = [];
    overlap_rect_index = 0;
    rect_length        = len(rectangles)
    
    for i1 in range(0, rect_length):
        
        #skip index of rectangle which label as overlap rectangle
        if i1 in overlap_list:
            continue
        
        rect_1 = rectangles[i1]
        
        for i2 in range(i1+1, rect_length):
            rect_2 = rectangles[i2]
            
            if check_overlap(rect_1, rect_2) == True:
            
                if calculate_iou_of(rect_1, rect_2) > threshold:
                    
                    #select the rectangle with lower confidence as overlap rectangle
                    if confidences[i1] >= confidences[i2]:
                        overlap_rect_index = i2
                    else:
                        overlap_rect_index = i1
                    
                    if overlap_rect_index not in overlap_list:
                        overlap_list.append(overlap_rect_index)

    return overlap_list

if __name__ == "__main__":
    i = 1
    ls = [1,2,3]
    rect1 = (10, 10, 60, 60)
    rect2 = (20, 20, 70, 70)
    
    iou = calculate_iou_of(rect1, rect2)
    
    print("iou = {}".format(iou))
    
    overlap_list = [1,2]
    list = [i for i in range(0, 5) if i not in overlap_list]
    print("ls = {}".format(list))
    
    if i not in ls:
        print("got it")
    else:
        print("not in")
    

    
    