# Import the os module, for the os.walk function
import fnmatch
import re
import os
import StringIO
import glob
import numpy as np
import re
import math 
import pylab
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker




  


p=0
i=1
dataRaw=[]
dataMod=[]
x=[]
y=[]
la=[]
s=[]

for files in glob.glob( '15datengefiltert.txt' ):
      with open( files, 'r' ) as f:
      # get date, 4th line of file
	f.readline()
	#f.readline()
	diameter0=f.readline()
	if ("Durchmesser" in diameter0):
	  diameterLineSplit = diameter0.split('\t')
	  durchmesser=diameter0.replace(',','.')
	print'',durchmesser[:]

      #dateLine = f.readline()
      #diameterlinesplit = diameterline.split('\t');
      #diameter= diameterlinesplit[1]
      #print'',diameterline











#for files in glob.glob( "15datengefiltert.txt" ):
    ##print'', files
    #s = open(files).read().replace(',','.')
    #dataRaw =  np.genfromtxt(StringIO.StringIO(s), skip_header = 4, skip_footer = 5)
    #diameter= dataRaw[-2,3]
    #dataMod = np.copy(dataRaw[5:,[1,2]]) # 1 = Zeit, 2 = Kraft
    ##dataMod.readline
    ##dataMod.readline
    ##dataMod.readline
    ##dataMod.readline
    ##x=dataMod[:,0] # 0 = Zeit entspricht der 2. Spalte in der Orginaldatei dataRaw[1]
    ##y=dataMod[:,1] # 1 = Kraft entspricht der 3. Spalte in der Orginaldatei dataRaw[2]
    #print'', diameter

    
    
    #la=plt.plot(x,y)
    #k=files[0:8]
    #print'',k
    #plt.savefig(str(k)+ ".png")    
    #p=p+1
    #plt.close
    #plt.clf() 
	   #dataRaw = np.genfromtxt(filenameMod)
	   #dataMod = np.copy(dataRaw[:,[0]])
	   #speicherdatei.close()
           #print f.name
       

 
