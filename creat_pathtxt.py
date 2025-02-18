import os
import re

#model-Train-val-test #
# dir=r'./data/image/HHD_高血压心脏病'#HHD_高血压心脏病,HCM_肥厚型心肌病
# fp = open('path/train_img_path_n8_g1.txt','a+') #a+追加，w+覆盖
# imgfile_list = os.listdir(r'./data/image/HHD_高血压心脏病')
# imgfile_list.sort(key= lambda x:str(x[:]))
# #print(img_list)
# seqsize =8
# gap=1
# for imgfile in imgfile_list[:30]:
#     filepath = os.path.join(dir,imgfile)
#     img_list = os.listdir(filepath)
#     img_list.sort(key=lambda x: int(re.findall('\d+',x)[0]))
#     #滑窗取序列
#     for i in range(0, len(img_list)-(seqsize*gap)+1, 1):
#         for j in range(i,i+(seqsize*gap),gap):
#               img = img_list[j]
#               path = os.path.join(filepath, img)
#               # if j == i+seqsize-1:
#               #     fp.write(path+'*')
#               # else:
#               #     fp.write(path+'*')
#               fp.write(path+'*')   
#         # for j in range(i,i+(seqsize*gap),gap):
#         #       img = img_list[j]
#         #       path = os.path.join(filepath, img)
#         #       fp.write('{}'.format(path.replace('image','label'))+'*')
#         #       # if j == i+seqsize-1:
#         #       #     fp.write(path+'*')
#         #       # else:
#         #       #     fp.write(path+'*')
#         #       #fp.write(path+'*')      
#         # fp.write('\n')
        
#         img = img_list[j-int(seqsize/2)*gap+1]
#         path = os.path.join(filepath, img)
#         fp.write('{}'.format(path.replace('image','label'))+'\n')
# fp.close()

#model-infer-otherframs#
dir=r'..//framing_data/HHD'#HHD_高血压心脏病,HCM_肥厚型心肌病
fp = open('path/all_img_path_n8_g1.txt','w+') #a+追加，w+覆盖
imgfile_list = os.listdir(r'..//framing_data/HHD')
imgfile_list.sort(key= lambda x:str(x[:]))
#print(img_list)
seqsize =8
gap=1
for imgfile in imgfile_list[:]:
    filepath = os.path.join(dir,imgfile)
    img_list = os.listdir(filepath)
    img_list.sort(key=lambda x: int(re.findall('\d+',x)[0]))
    #滑窗取序列
    for i in range(0, len(img_list)-(seqsize*gap)+1, 1):
        for j in range(i,i+(seqsize*gap),gap):
              img = img_list[j]
              path = os.path.join(filepath, img)
              # if j == i+seqsize-1:
              #     fp.write(path+'*')
              # else:
              #     fp.write(path+'*')
              fp.write(path+'*')   
        # for j in range(i,i+(seqsize*gap),gap):
        #       img = img_list[j]
        #       path = os.path.join(filepath, img)
        #       fp.write('{}'.format(path.replace('image','label'))+'*')
        #       # if j == i+seqsize-1:
        #       #     fp.write(path+'*')
        #       # else:
        #       #     fp.write(path+'*')
        #       #fp.write(path+'*')      
        # fp.write('\n')
        
        # img = img_list[j-int(seqsize/2)*gap+1]
        # path = os.path.join(filepath, img)
        #fp.write('{}'.format(path.replace('image','label'))+'\n')
fp.close()
