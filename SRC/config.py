''' Files '''
input_video = 'Data/Video.mp4'
output_video = "Data/Barca1.avi"
output_map = "Data/map.avi"
output_warp = "/Data/warp.avi"

''' Tracking parameters ''' 
# Number of tracked position for the ball
num_tracked_points = 25


''' Object Colors parameters ''' 
# get a different color array for each of the classes
COLORS = {
  "0": (255,0,0), # BARCA [Blue]
  "1": (255,255,255), # REAL [White]
  "2": (0,255,255), # BALL [Yellow]
  "3": (0,0,0) # BALL [Black]
}


''' Model parameters ''' 
confThreshold = 0.85  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image


'''YOLOV4 Model Information '''
# Model Information 
model_configurations = 'Model/football.cfg'
model_weights = 'Model/football_best.weights'


''' Deep Homo Model Information'''
DeepHomoModel_weights = "https://storage.googleapis.com/narya-bucket-1/models/deep_homo_model.h5"
WEIGHTS_NAME = "deep_homo_model.h5"
WEIGHTS_TOTAR = False

''' Map Draw Configurations '''
map_location = 'Data/Map/world_cup_template.png' # Image used
scale_percent = 12 # Percentage to scale image down

y_offset = 850
x_offset = 800 # 800

map_alpha = 0.7
map_beta = 0.5
map_gamma = 0.5

''' Players Map Configurations '''
OldMax_x = 1920
OldMin_x = 0
NewMax_x = 122 # 245 (Original)
NewMin_x = 0

old_range_x = (OldMax_x - OldMin_x)  
new_range_x = (NewMax_x - NewMin_x)  


OldMax_y = 1080
OldMin_y = 0
NewMax_y = 162
NewMin_y = 0

old_range_y = (OldMax_y - OldMin_y)  
new_range_y = (NewMax_y - NewMin_y)  

'''Image Show resize configurations '''
resized_width = 960
resized_height = 540