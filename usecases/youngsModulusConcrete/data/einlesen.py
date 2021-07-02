# Import the os module, for the os.walk function
import fnmatch
import re
import os
import glob
import numpy as np
import re


  
k= os.getcwd()


p=0
i=1

for files in glob.glob( "*.txt" ):
    with open( files, 'r' ) as f:
      # get date, 4th line of file
      #f.readline()
      #f.readline()
      #f.readline()
      #dateLine = f.readline()
      #dateLineSplit = dateLine.split('\t');
      #date = dateLineSplit[0]
	  
      #f.seek(0) # jump to first line
      file_contents = f.read()
      #f.seek(0)
      #print f.readline()
      #print len(dateLine.split('\t'))
      if ("Kraft Soll" in file_contents):
	      
	#dateLineSplit = dateLine.split('\t');
	#date = dateLineSplit[1]

	   #for line in file_contents:
	    #if line.startswith("Datum"):


	     
	speicherdatei = open(str(p) + "datengefiltert"  ".txt", "w")
	speicherdatei.write(file_contents)
	   #dataRaw = np.genfromtxt(filenameMod)
	   #dataMod = np.copy(dataRaw[:,[0]])
	   #speicherdatei.close()
           #print f.name
	p=p+1

print '', p 