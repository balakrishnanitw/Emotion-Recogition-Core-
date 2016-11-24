# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:11:18 2016

@author: pankaj

"""
import os
import numpy as np
from numpy.fft import fft, ifft
import warnings
from numpy import real
from scipy.io.wavfile import read
from sklearn import preprocessing
from numpy import mean, sqrt, square
from scipy.signal import lfilter, hamming
from scipy.fftpack.realtransforms import dct

def findmean(x):
    """"
    finding out the mean feature of all the samples
    
    """
    return np.mean(x)

def findmaxima(x):
    """
    maxima of each samples

    """ 
    return x.max()# maxima.reshape(-1,1) 

def findminima(x):
    """
    minima of each samples 

    """
    return x.min()#min_f.reshape(-1,1)    

def findrms(x):
    """
    root man square energy of the samples    
    
    """
    return sqrt(mean(square(x)))#rms_f.reshape(-1,1)
    
def findzcr(x):
    """
    zero crossing rate of the SAMPLES 

    """     
    return (np.diff(np.sign(x)) != 0).sum()#zcr_f.reshape(-1,1)     
   

def findlpc(x):
    """"
    
    12 lpc coeffiecients are determined using following
    equation
    """
    LPCcoeff,Err,Ka = lpc(x,11)
    k=LPCcoeff[1:11]
    lpc12 = np.asarray(k)
    return lpc12

def lpc(x, N):
    """Linear Predictor Coefficients.

    :param x:
    :param int N: default is length(X) - 1
    
    :Details:

    Finds the coefficients :math:`A=(1, a(2), \dots a(N+1))`, of an Nth order 
    forward linear predictor that predicts the current value value of the 
    real-valued time series x based on past samples:
    
    .. math:: \hat{x}(n) = -a(2)*x(n-1) - a(3)*x(n-2) - ... - a(N+1)*x(n-N)

    such that the sum of the squares of the errors

    .. math:: err(n) = X(n) - Xp(n)

    is minimized. This function  uses the Levinson-Durbin recursion to 
    solve the normal equations that arise from the least-squares formulation.  

    .. seealso:: :func:`levinson`, :func:`aryule`, :func:`prony`, :func:`stmcb`

    .. todo:: matrix case, references
    
    :Example:

    ::
    
        from scipy.signal import lfilter
        noise = randn(50000,1);  % Normalized white Gaussian noise
        x = filter([1], [1 1/2 1/3 1/4], noise)
        x = x[45904:50000]
        x.reshape(4096, 1)
    
        x = x[0]

    Compute the predictor coefficients, estimated signal, prediction error, and autocorrelation sequence of the prediction error:
   

    1.00000 + 0.00000i   0.51711 - 0.00000i   0.33908 - 0.00000i   0.24410 - 0.00000i

    ::
 
        a = lpc(x, 3)
        est_x = lfilter([0 -a(2:end)],1,x);    % Estimated signal
        e = x - est_x;                        % Prediction error
        [acs,lags] = xcorr(e,'coeff');   % ACS of prediction error

    
    a = lpc(signal2array, 3)
    """
    m = len(x)    
#    if N == None:
#        N = m - 1 #default value if N is not provided
#    elif N > m-1:
#        #disp('Warning: zero-padding short input sequence')
#        signal2array.resize(N+1)
#        #todo: check this zero-padding. 
    x = x + 0.001*np.ones_like(x)
#    X = fft(x)
    X = fft(x, 2**nextpow2(2.*len(x)-1))
    R = real(ifft(abs(X)**2))
    R = R/(m-1.) #Biased autocorrelation estimate
    return levinson_1d(R, N)

def nextpow2(n):
    """Return the next power of 2 such as 2^p >= n.
    Notes
    -----
    Infinite and nan are left untouched, negative values are not allowed."""
    if np.any(n < 0):
        raise ValueError("n should be > 0")

    if np.isscalar(n):
        f, p = np.frexp(n)
        if f == 0.5:
            return p-1
        elif np.isfinite(f):
            return p
        else:
            return f
    else:
        f, p = np.frexp(n)
        res = f
        bet = np.isfinite(f)
        exa = (f == 0.5)
        res[bet] = p[bet]
        res[exa] = p[exa] - 1
        return res

def levinson_1d(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.

    Parameters
    ---------
    r : array-like
        input array to invert (since the matrix is symmetric Toeplitz, the
        corresponding pxp matrix is defined by p items only). Generally the
        autocorrelation of the signal for linear prediction coefficients
        estimation. The first item must be a non zero real.

    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation:

                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :      *  :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
                       _
    with respect to a (  is the complex conjugate). Using the special symmetry
    in the matrix, the inversion can be done in O(p^2) instead of O(p^3).
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if n < 1:
        raise ValueError("Cannot operate on empty array !")
    elif order > n - 1:
        raise ValueError("Order should be <= size-1")

    if not np.isreal(r[0]):
        raise ValueError("First item of input must be real.")
    elif not np.isfinite(1/r[0]):
        raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order+1, r.dtype)
    # temporary array
    t = np.empty(order+1, r.dtype)
    # Reflection coefficients
    k = np.empty(order, r.dtype)

    a[0] = 1.
    e = r[0]

    for i in xrange(1, order+1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k[i-1] = -acc / e
        a[i] = k[i-1]

        for j in range(order):
            t[j] = a[j]
           # print a
        for j in range(1, i):
            a[j] += k[i-1] * np.conj(t[i-j])

        e *= 1 - k[i-1] * np.conj(k[i-1])

    return a, e, k


def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.
    example:
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])
    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:
            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value
    endvalue    The value to use for end='pad'
    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').
    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length:
        raise ValueError, "frames cannot overlap by more than 100%"
    if overlap < 0 or length <= 0:
        raise ValueError, "overlap must be nonnegative and length must "\
                          "be positive"

    if l < length or (l-length) % (length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length-overlap) \
               or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1,axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s,dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l == 0:
        raise ValueError, \
              "Not enough data points to segment array in 'cut' mode; "\
              "try 'pad' or 'wrap'"
    assert l >= length
    assert (l-length) % (length-overlap) == 0
    n = 1 + (l-length) // (length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s,s) + a.strides[axis+1:]

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s,s) \
                     + a.strides[axis+1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    
def trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
    """Compute triangular filterbank for MFCC computation."""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank, freqs

def mfcc(s, nwin=256, nfft=512, fs=48000, nceps=13):
    """Compute Mel Frequency Cepstral Coefficients.
    Parameters
    ----------
    input: ndarray
        input from which the coefficients are computed
    Returns
    -------
    ceps: ndarray
        Mel-cepstrum coefficients
    mspec: ndarray
        Log-spectrum in the mel-domain.
    Notes
    -----
    MFCC are computed as follows:
        * Pre-processing in time-domain (pre-emphasizing)
        * Compute the spectrum amplitude by windowing with a Hamming window
        * Filter the signal in the spectral domain with a triangular
        filter-bank, whose filters are approximatively linearly spaced on the
        mel scale, and have equal bandwith in the mel scale
        * Compute the DCT of the log-spectrum
    References
    ----------
    .. [1] S.B. Davis and P. Mermelstein, "Comparison of parametric
           representations for monosyllabic word recognition in continuously
           spoken sentences", IEEE Trans. Acoustics. Speech, Signal Proc.
           ASSP-28 (4): 357-366, August 1980."""

    # MFCC parameters: taken from auditory toolbox
    over = nwin - 160
    # Pre-emphasis factor (to take into account the -6dB/octave rolloff of the
    # radiation at the lips level)
    prefac = 0.97

    #lowfreq = 400 / 3.
    lowfreq = 133.33
    #highfreq = 6855.4976
    linsc = 200/3.
    logsc = 1.0711703

    nlinfil = 13
    nlogfil = 27
    #nfil = nlinfil + nlogfil

    w = hamming(nwin, sym=0)

    fbank = trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)[0]

    #------------------
    # Compute the MFCC
    #------------------
    extract = preemp(s, prefac)
    framed = segment_axis(extract, nwin, over) * w
    
    # Compute the spectrum magnitude
    spec1 = np.abs(fft(framed, nfft, axis=-1))
    spec = spec1 + 0.001*np.ones_like(spec1) #To avoid -Inf and nan
    # Filter the spectrum through the triangle filterbank
    mspec = np.log10(np.dot(spec, fbank.T))
    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, :nceps]
    ceps = ceps.transpose()
    U, s, Vh = np.linalg.svd(ceps)        

    return np.asarray(U).reshape(-1)

def preemp(s, p):
    """Pre-emphasis filter."""
    return lfilter([1., -p], 1, s)   
    
if __name__ == "__main__":
#    LPC_all = []
    wavs = []
    feature = []
    src_dir = "C:\\Users\\IBM_ADMIN\\Documents\\Python Scripts\\TestedFeature\\audio\\Mixed Train"
    for file in os.listdir(src_dir):
        if file.endswith(".wav"):
            wavs.append(read(src_dir+'\\'+file))
            signal = read(src_dir+'\\'+file)
            signal2array= np.array(signal[1], dtype=float)   
            framed_signal2array = segment_axis(signal2array, 256, 0)
#            LPC_all.append(findlpc(signal2array))
#            lpc_f = np.asarray(LPC_all)
            framed_feature_all = []
            for i in xrange(0,len(framed_signal2array)):
                
                framed_mean = findmean(framed_signal2array[i]).reshape(-1,1)
                framed_maxima = findmaxima(framed_signal2array[i]).reshape(-1,1)
                framed_minima = findminima(framed_signal2array[i]).reshape(-1,1)
                framed_rms = findrms(framed_signal2array[i]).reshape(-1,1)
                framed_zcr = findzcr(framed_signal2array[i]).reshape(-1,1)
                framed_LPC = findlpc(framed_signal2array[i]).reshape(-1,1)
                framed_MFCC = mfcc(framed_signal2array[i]).reshape(-1,1)
                framed_feature = (np.concatenate((framed_mean,framed_maxima,\
                framed_minima,framed_rms,framed_zcr,framed_LPC,framed_MFCC),axis=0)).T
                
                framed_feature_all.append(framed_feature)
                X = np.asarray(framed_feature_all)
                X_data = X[:,0,:]
                Y = np.mean(X_data, axis=0).reshape(1,-1)
                
            feature.append(Y)            

Z = np.asarray(feature) 
Z = Z[:,0,:]               
scaler = preprocessing.StandardScaler().fit(Z)
normalized_mean1 = scaler.mean_ 
normalized_mean = normalized_mean1.reshape(1,184)
normalized_scale1 = scaler.scale_
normalized_scale = normalized_scale1.reshape(1,184)
NormalizedFeature = scaler.transform(Z)          

#SAVE NORMALIZED FEATURE!!

#%% Writing the fitted parameters to Json Male

import json

norm_mean_list  = normalized_mean.tolist()
norm_scale_list = normalized_scale.tolist()

parameters = {'mean' : norm_mean_list,'scale': norm_scale_list }

# Write to json file
with open('parameters.json', 'w') as fp:
    json.dump(parameters, fp, sort_keys=True, indent=4)
    
#%% Writing the fitted parameters to Json Female

import json

norm_mean_list  = normalized_mean.tolist()
norm_scale_list = normalized_scale.tolist()

parameters = {'mean' : norm_mean_list,'scale': norm_scale_list }

# Write to json file
with open('parametersFemale.json', 'w') as fp:
    json.dump(parameters, fp, sort_keys=True, indent=4)
#%% Writing the fitted parameters to Json Complete 

import json

norm_mean_list  = normalized_mean.tolist()
norm_scale_list = normalized_scale.tolist()

parameters = {'mean' : norm_mean_list,'scale': norm_scale_list }

# Write to json file
with open('parametersComplete.json', 'w') as fp:
    json.dump(parameters, fp, sort_keys=True, indent=4)
