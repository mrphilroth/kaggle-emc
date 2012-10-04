# Read data stored as a CSR matrix and labels
from scipy import sparse
import numpy

def EMC_ReadData(filePath) :
    """ USAGE:
        reads a CSR sparse matrix from file and converts it
        to a Matrix library object in a CSR format
    
        PARAMETERS:
        filePath - full/relative path to the data file
    """
    
    # open file for reading
    inFile = open(filePath, "r")

    # read matrix shape
    matrixShape = numpy.fromstring(inFile.readline(), dtype = 'int', sep = ',');

    # read matrix data, indices and indptr
    data = numpy.fromstring(inFile.readline(), dtype = 'float', sep = ',');
    indices = numpy.fromstring(inFile.readline(), dtype = 'int', sep = ',');
    indptr = numpy.fromstring(inFile.readline(), dtype = 'int', sep = ',');

    # close file
    inFile.close()

    return sparse.csr_matrix((data, indices, indptr),
                             shape = (matrixShape[0], matrixShape[1]))

def EMC_ReadLabels( filePath ):
    """ USAGE:
        reads a list of labels into memory
    
        PARAMETERS:
        filePath - full/relative path to the data file
    
        RETURN:
        list of labels
     """

    # read data from file
    data = numpy.loadtxt(open(filePath, "r"), dtype='int', delimiter = ",")
    return data

