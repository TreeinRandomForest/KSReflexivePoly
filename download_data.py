import sys
import os
import requests
from bs4 import BeautifulSoup

sess = requests.Session() #open a session to make multiple calls
D3_URL = 'http://hep.itp.tuwien.ac.at/~kreuzer/pub/K3/RefPoly.d3'
D4_URL = 'http://quark.itp.tuwien.ac.at/~kreuzer/V/'

def get_d3_data(D3_URL, TARGETDIR):
    resp = sess.get(D3_URL)
    result = resp.text

    dirname = '{}/d3/'.format(TARGETDIR)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    output_filename = "{}/{}".format(dirname, D3_URL.split("/")[-1])
    with open(output_filename, 'w') as output_f:
        print(result, file=output_f)
        
def validate_d3_data():
    pass

def get_d4_data(D4_URL, TARGETDIR):
    pass

def validate_d4_data():
    pass
        
if __name__ == "__main__":
    #extract arguments
    try:
        TARGETDIR = sys.argv[1] #place to store data
        DIM = int(sys.argv[2]) #3 or 4 for K3/CY data
    except:
        print("Usage: python download_data.py [TARGETDIR] [DIM=3/4]")
        print("Example: python download_data.py data 3")
        sys.exit(-1)

    #download data
    if DIM==3:
        get_d3_data(D3_URL, TARGETDIR)
        validate_d3_data()
    elif DIM==4:
        get_d4_data(D4_URL)
        validate_d4_data()
    else:
        print("Second argument (DIM) should be 3 or 4")
