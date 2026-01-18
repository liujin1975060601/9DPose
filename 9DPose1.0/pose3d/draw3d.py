import cv2

def draw_proj_box3d(K, R, t, abc, dst, color, thick):
    box = [ -abc[0] / 2, abc[0] / 2, -abc[1] / 2, abc[1] / 2, -abc[2] / 2, abc[2] / 2 ]
    box_pts[0] = cv2.Point3f(box[0], box[2], box[4])
	box_pts[1] = cv2.Point3f(box[1], box[2], box[4])
	box_pts[2] = cv2.Point3f(box[1], box[3], box[4])
	box_pts[3] = cv2.Point3f(box[0], box[3], box[4])
	box_pts[4] = cv2.Point3f(box[0], box[2], box[5])
	box_pts[5] = cv2.Point3f(box[1], box[2], box[5])
	box_pts[6] = cv2.Point3f(box[1], box[3], box[5])
	box_pts[7] = cv2.Point3f(box[0], box[3], box[5])
                 
    pts_proj = cv2.projectPoints(box_pts, R64, t64, K)



                                                                                                               
   
                                        
                                                                                              
                                                                                                
                                                    
                                                    
                                                    
                                                    
                                                    
                                                    
                                                    
                                                    
     
                                  
               
                              
     
               
                              
     
                                     
                                                                
                                
                                                                                                                       
                         
                                                                                           
                                                         
                                                         
                                                                                                                       
                                                                                
      
                                                         
                                                         
                                                         
                                                         
     
                                                                                                                       
                                                                                
                                

