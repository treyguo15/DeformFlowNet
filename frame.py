import cv2
 
def video2frame(videos_path,frames_save_path,time_interval):
 
  '''
  :param videos_path: 视频的存放路径
  :param frames_save_path: 视频切分成帧之后图片的保存路径
  :param time_interval: 保存间隔
  :return:
  '''
  vidcap = cv2.VideoCapture(videos_path)
  success, image = vidcap.read()
  #print(image.shape)
  count = 0
  while success:
    success, image = vidcap.read()
    count += 1
    #image1=image[0:460,80:555]       #432*636
    #image1=image[100:600,220:755]  
    #image1=image[150:650,220:755]     #708*1016
    #image1=image[130:590,180:655]  #philips
    image1=image[100:560,170:645]  #philips无logo600*800
    if count % time_interval == 0:
      cv2.imencode('.jpg', image1)[1].tofile(frames_save_path + "/%d.jpg" % count)
    # if count == 20:
    #   break
  #print(count)
 
if __name__ == '__main__':
   videos_path = '/media//Research/xz/xx.avi'
   frames_save_path = '/media/Research/xz/'
   time_interval = 1#隔一帧保存一次
   video2frame(videos_path, frames_save_path, time_interval) 
