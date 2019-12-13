import sys, os, json
import cv2

imgdir = sys.argv[1]
resfile = sys.argv[2]
with open(resfile, 'r') as f:
    anns = json.load(f)
cnt = 0
for k, v in anns.items():
    cnt += 1
    if cnt >= 50:
        break
    #imname="%05d.jpg"%int(k)
    imname = k
    impath = os.path.join(imgdir, imname)
    img = cv2.imread(impath)

    for b in v:
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0,0,255), 2)
    cv2.imshow('res', img)
    cv2.waitKey()
