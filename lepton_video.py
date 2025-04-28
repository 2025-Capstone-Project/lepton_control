import sys
import numpy as np
import cv2
from pylepton.Lepton3 import Lepton3
from datetime import datetime
import time

MIN_TEMP = 10 + 273.15
MAX_TEMP = 60 + 273.15

def capture(flip_v = False, device = "/dev/spidev0.0"):
  with Lepton3(device) as l:
    #l.set_agc_enable(0)
    a,_ = l.capture()
    #np.right_shift(a, 8, scaled)
    min_adc = MIN_TEMP * 100
    max_adc = MAX_TEMP * 100
    temp_k = a / 100.0
    temp_c = temp_k - 273.15
    a[a<=min_adc] = min_adc + 1
    a[a>=max_adc] = max_adc - 1
    scaled = np.clip((a - min_adc) / (max_adc - min_adc), 0, 1)

  if flip_v:
    cv2.flip(a,0,a)
  cv2.normalize(scaled, None, alpha = 0, beta = 65535, norm_type = cv2.NORM_MINMAX)
  #np.right_shift(a, 8, a)
  scaled = scaled * 256
  


  print(f" 최소 온도: {np.min(temp_c):.2f} C / {np.min(temp_k): .2f} K")
  print(f" 최대 온도: {np.max(temp_c):.2f} C / {np.max(temp_k): .2f} K")

  return np.uint8(scaled), np.min(temp_c), np.max(temp_c)

if __name__ == '__main__':
  from optparse import OptionParser

  a = 0
  
  usage = "usage: %prog [options] output_file[.format]"
  parser = OptionParser(usage=usage)

  parser.add_option("-f", "--flip-vertical",
                    action="store_true", dest="flip_v", default=False,
                    help="flip the output image vertically")

  parser.add_option("-d", "--device",
                    dest="device", default="/dev/spidev0.0",
                    help="specify the spi device node (might be /dev/spidev0.1 on a newer device)")

  (options, args) = parser.parse_args()

  if len(args) < 1:
    print("You must specify an output filename")
    sys.exit(1)

  while (a <= int(args[0])):
    image, min_temp, max_temp = capture(flip_v = options.flip_v, device = options.device)
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    img_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " min_" + str(min_temp) + " max_" + str(max_temp) + ".png"
    cv2.imshow("video", image)
    a+=1
    cv2.waitKey(33)
